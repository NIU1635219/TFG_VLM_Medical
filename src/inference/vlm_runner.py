"""Módulo de carga e inferencia VLM sobre LM Studio usando lmstudio-python."""

import base64
import io
import json
import dataclasses
import math
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, Type, TypeVar, cast, overload

from pydantic import BaseModel, ValidationError

from src.inference.schemas import (
    GenericObjectDetection,
    PolypDetection,
    SycophancyTest,
    ImageQualityAssessment,
    SCHEMA_REGISTRY,  # noqa: F401
)

# TypeVar genérico ligado a BaseModel para que el IDE infiera el tipo de retorno de inference()
T = TypeVar("T", bound=BaseModel)

# Alias de retro-compatibilidad (usado por tests y scripts heredados)
VLMStructuredResponse = GenericObjectDetection


@dataclasses.dataclass(frozen=True)
class InferenceTelemetry:
    """Métricas de rendimiento extraídas de una inferencia individual."""

    total_duration_seconds: float
    ttft_seconds: float | None
    generation_duration_seconds: float | None
    prompt_tokens_per_second: float | None
    tokens_per_second: float | None
    reasoning_tokens: int | None
    stop_reason: str | None
    model_id: str | None
    vram_usage_mb: float | None
    total_params: int | str | None
    architecture: str | None
    quantization: str | None
    gpu_layers: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    stats: dict[str, Any]
    resources: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class InferenceResult(Generic[T]):
    """Respuesta tipada del modelo junto con sus métricas de telemetría."""

    data: T
    telemetry: InferenceTelemetry

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import lmstudio as lms
except ImportError:
    lms = None


class _LMSModelHandle(Protocol):
    def respond(self, history: Any, **kwargs: Any) -> Any:
        ...


class _LMSChatHandle(Protocol):
    def add_user_message(self, content: Any, *, images: Any = (), _files: Any = ()) -> Any:
        ...


class VLMLoader:
    """
    Gestor de carga e inferencia para modelos VLM (Vision-Language Models) en LM Studio.
    
    Esta clase facilita la interacción con la API de LM Studio para cargar modelos,
    enviar imágenes y preguntas, y obtener respuestas estructuradas.
    
    Attributes:
        model_tag (str): Identificador o ruta del modelo a cargar.
        verbose (bool): Si es True, imprime información de depuración.
        server_api_host (str | None): URL del servidor de la API, si es diferente al local.
        api_token (str | None): Token de autenticación para la API, si es necesario.
    """

    def __init__(
        self,
        model_path: str,
        verbose: bool = False,
        server_api_host: str | None = None,
        api_token: str | None = None,
    ):
        """
        Inicializa el cargador de VLM.

        Args:
            model_path (str): El identificador del modelo en LM Studio (ej. "publisher/repo").
            verbose (bool, optional): Habilita logs detallados. Por defecto es False.
            server_api_host (str, optional): Host de la API de LM Studio.
            api_token (str, optional): Token de API para autenticación.
            
        Raises:
            ValueError: Si `model_path` está vacío o parámetros opcionales son cadenas vacías.
        """
        if not model_path or not str(model_path).strip():
            raise ValueError("model_path no puede estar vacío")

        if server_api_host is not None and not str(server_api_host).strip():
            raise ValueError("server_api_host no puede ser vacío")

        if api_token is not None and not str(api_token).strip():
            raise ValueError("api_token no puede ser vacío")

        self.model_tag = str(model_path).strip()
        self.verbose = bool(verbose)
        self.server_api_host = str(server_api_host).strip() if server_api_host else None
        self.api_token = str(api_token).strip() if api_token else None
        self._client: Any | None = None
        self._loaded_model: _LMSModelHandle | None = None

    def _create_lms_client(self) -> Any:
        """
        Crea e inicializa una instancia del cliente de LM Studio.
        
        Configura el cliente con el host y token si fueron proporcionados.
        
        Returns:
            Any: Instancia configurada de `lms.Client`.
            
        Raises:
            RuntimeError: Si la clase Client no está disponible en el paquete lms.
        """
        assert lms is not None
        client_cls = getattr(lms, "Client", None)
        if not callable(client_cls):
            raise RuntimeError("Client API no disponible")

        kwargs: dict[str, Any] = {}
        if self.api_token:
            kwargs["api_token"] = self.api_token

        try:
            if self.server_api_host:
                return client_cls(self.server_api_host, **kwargs)
            return client_cls(**kwargs)
        except Exception as error:
            if self._is_lmstudio_connection_issue(error):
                raise RuntimeError(
                    self._build_lmstudio_connection_message("crear el cliente de LM Studio", error)
                ) from error
            raise

    def _ensure_lms_available(self) -> None:
        """
        Verifica que la librería `lmstudio-python` esté instalada y disponible.
        
        Raises:
            RuntimeError: Si la librería no se pudo importar.
        """
        if lms is None:
            raise RuntimeError(
                "La librería 'lmstudio-python' no está instalada. "
                "Instala y activa LM Studio SDK para continuar."
            )

    def _is_lmstudio_connection_issue(self, error: BaseException) -> bool:
        """
        Detecta si una excepción apunta a un servidor LM Studio inaccesible.

        Args:
            error (BaseException): Error capturado durante una operación con el SDK.

        Returns:
            bool: ``True`` si el mensaje parece corresponder a un problema de
            conexión con el servidor LM Studio.
        """
        error_text = str(error).lower()
        patterns = (
            "connection refused",
            "actively refused",
            "failed to establish a new connection",
            "max retries exceeded",
            "could not connect",
            "cannot connect",
            "server is not running",
            "winerror 10061",
            "errno 111",
        )
        return any(pattern in error_text for pattern in patterns)

    def _build_lmstudio_connection_message(self, operation: str, error: BaseException) -> str:
        """
        Genera un mensaje claro cuando LM Studio no está accesible.

        Args:
            operation (str): Acción que se estaba ejecutando.
            error (BaseException): Error original detectado.

        Returns:
            str: Mensaje listo para elevar al usuario final.
        """
        target_host = self.server_api_host or "localhost:1234"
        return (
            f"No se pudo {operation} porque LM Studio no está accesible en {target_host}. "
            "Verifica que la aplicación esté abierta, que el servidor local esté encendido "
            "y que el host configurado sea correcto. "
            f"Detalle original: {error}"
        )

    def _extract_model_key(self, model_obj: Any) -> str | None:
        """
        Extrae el identificador único (key) de un objeto de modelo de LM Studio.
        
        Intenta obtener la clave desde varias fuentes: string directo, diccionario
        o atributo de objeto.
        
        Args:
            model_obj (Any): El objeto del cual extraer la clave.
            
        Returns:
            str | None: La clave del modelo o None si no se encuentra.
        """
        if model_obj is None:
            return None

        if isinstance(model_obj, str):
            text = model_obj.strip()
            return text or None

        if isinstance(model_obj, dict):
            for key_name in ("key", "id", "model"):
                value = model_obj.get(key_name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        for attr_name in ("key", "id", "model"):
            value = getattr(model_obj, attr_name, None)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return None

    def _list_loaded_models_keys(self) -> list[str]:
        """
        Obtiene una lista de las claves de los modelos actualmente cargados en LM Studio.
        
        Realiza una llamada segura a `list_loaded_models` del SDK y procesa
        recursivamente la respuesta para encontrar identificadores válidos.
        
        Returns:
            list[str]: Lista de claves de modelos cargados. Devuelve lista vacía en caso de error.
        """
        try:
            self._ensure_lms_available()
            list_fn = getattr(lms, "list_loaded_models", None)
            if not callable(list_fn):
                return []

            response = list_fn()
            keys: list[str] = []

            def collect(items: Any) -> None:
                if items is None:
                    return
                if isinstance(items, (list, tuple)):
                    for item in items:
                        model_key = self._extract_model_key(item)
                        if model_key:
                            keys.append(model_key)
                    return
                if isinstance(items, dict):
                    if "data" in items:
                        collect(items.get("data"))
                        return
                    if "models" in items:
                        collect(items.get("models"))
                        return
                    model_key = self._extract_model_key(items)
                    if model_key:
                        keys.append(model_key)
                    return

                model_key = self._extract_model_key(items)
                if model_key:
                    keys.append(model_key)

                data_attr = getattr(items, "data", None)
                if data_attr is not None:
                    collect(data_attr)
                models_attr = getattr(items, "models", None)
                if models_attr is not None:
                    collect(models_attr)

            collect(response)
            return list(dict.fromkeys(keys))
        except Exception:
            return []

    def _build_structured_instruction(
        self,
        prompt: str,
        schema: type[BaseModel] | None = None,
    ) -> str:
        """
        Construye el prompt final añadiendo instrucciones para forzar la salida JSON.

        Si se proporciona un esquema Pydantic, extrae dinámicamente los nombres de
        los campos y sus descripciones para generar instrucciones precisas.
        Si no se proporciona esquema, usa las claves de ``GenericObjectDetection``.

        Args:
            prompt (str): La instrucción original del usuario.
            schema (type[BaseModel] | None): Clase Pydantic objetivo. Si es None
                se utiliza ``GenericObjectDetection`` como contrato por defecto.

        Returns:
            str: El prompt modificado con instrucciones de formato JSON estricto.
        """
        target_schema = schema if schema is not None else GenericObjectDetection
        prompt_clean = prompt.strip()

        # Genera la lista de campos desde el esquema Pydantic de forma dinámica
        field_lines_parts: list[str] = []
        for field_name, field_info in target_schema.model_fields.items():
            desc = ""
            if field_info.description:
                # Recorta la descripción a 80 caracteres para no saturar el contexto
                desc = f" – {field_info.description[:80]}"
            field_lines_parts.append(f"- {field_name}{desc}")

        field_lines = "\n".join(field_lines_parts)

        return (
            f"{prompt_clean}\n\n"
            "Responde EXCLUSIVAMENTE con JSON válido, sin texto adicional, "
            "con estas claves exactas:\n"
            f"{field_lines}\n"
            "No incluyas markdown, comentarios ni explicaciones fuera del JSON."
        )

    def _extract_response_text(self, response: Any) -> str:
        """
        Extrae el contenido textual de la respuesta del modelo, manejando diversos formatos.
        
        Procesa respuestas que pueden ser cadenas, objetos con atributos o diccionarios anidados.
        
        Args:
            response (Any): El objeto de respuesta devuelto por el modelo.
            
        Returns:
            str: El texto extraído de la respuesta.
        """

        def extract(value: Any) -> str:
            if value is None:
                return ""


            if isinstance(value, str):
                return value

            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    text_item = extract(item)
                    if text_item:
                        parts.append(text_item)
                return "\n".join(parts).strip()

            if isinstance(value, dict):
                for key_name in ("text", "content", "response", "output"):
                    if key_name in value:
                        text_value = extract(value.get(key_name))
                        if text_value:
                            return text_value

                message = value.get("message")
                message_text = extract(message)
                if message_text:
                    return message_text

                choices = value.get("choices")
                if isinstance(choices, list) and choices:
                    choice_text = extract(choices[0])
                    if choice_text:
                        return choice_text

                return ""

            for attr_name in ("text", "content", "response", "output"):
                attr_value = getattr(value, attr_name, None)
                text_value = extract(attr_value)
                if text_value:
                    return text_value

            message_attr = getattr(value, "message", None)
            message_text = extract(message_attr)
            if message_text:
                return message_text

            choices_attr = getattr(value, "choices", None)
            if isinstance(choices_attr, list) and choices_attr:
                return extract(choices_attr[0])

            return ""

        return extract(response).strip()

    def _object_to_dict(self, value: Any) -> dict[str, Any]:
        """Convierte objetos del SDK o respuestas REST a un diccionario serializable."""
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)

        if not isinstance(value, type) and dataclasses.is_dataclass(value):
            try:
                dumped = dataclasses.asdict(cast(Any, value))
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, dict):
                return dumped

        to_dict = getattr(value, "dict", None)
        if callable(to_dict):
            dumped = to_dict()
            if isinstance(dumped, dict):
                return dumped

        value_dict = getattr(value, "__dict__", None)
        if isinstance(value_dict, dict):
            return {key: nested for key, nested in value_dict.items() if not key.startswith("_")}

        collected: dict[str, Any] = {}
        for attr_name in dir(value):
            if attr_name.startswith("_"):
                continue
            try:
                attr_value = getattr(value, attr_name)
            except Exception:
                continue
            if callable(attr_value):
                continue
            if isinstance(attr_value, Path):
                collected[attr_name] = str(attr_value)
            elif isinstance(attr_value, (str, int, float, bool, type(None), dict, list, tuple, set)):
                collected[attr_name] = attr_value

        if collected:
            return collected

        return {}

    def _extract_numeric_value(self, payload: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
        """Extrae un valor numérico desde varios alias posibles."""
        for alias in aliases:
            if alias not in payload:
                continue
            value = payload.get(alias)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value.strip())
                except ValueError:
                    continue
        return None

    def _extract_string_value(self, payload: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
        """Extrae un valor de texto desde varios alias posibles."""
        for alias in aliases:
            value = payload.get(alias)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized
        return None

    def _estimate_reasoning_tokens(self, validated: BaseModel) -> int | None:
        """Cuenta tokens aproximados a partir del campo ``reasoning`` del schema validado."""
        reasoning_value = getattr(validated, "reasoning", None)
        if not isinstance(reasoning_value, str):
            return None

        normalized_reasoning = reasoning_value.strip()
        if not normalized_reasoning:
            return 0

        return len(re.findall(r"\S+", normalized_reasoning))

    def _extract_inference_telemetry(self, response: Any, wall_duration_seconds: float) -> InferenceTelemetry:
        """Extrae telemetría fiable desde `response.stats` y `/v1/models`."""
        stats_payload = self._object_to_dict(getattr(response, "stats", None))
        if not stats_payload:
            result_payload = self._object_to_dict(getattr(response, "result", None))
            stats_payload = self._object_to_dict(result_payload.get("stats")) if result_payload else {}
        model_info_payload = self._object_to_dict(getattr(response, "model_info", None))

        prompt_tokens_value = self._extract_numeric_value(
            stats_payload,
            (
                "prompt_tokens_count",
            ),
        )
        completion_tokens_value = self._extract_numeric_value(
            stats_payload,
            (
                "predicted_tokens_count",
            ),
        )
        tokens_per_second = self._extract_numeric_value(
            stats_payload,
            (
                "tokens_per_second",
            ),
        )
        ttft_seconds = self._extract_numeric_value(
            stats_payload,
            (
                "time_to_first_token_sec",
            ),
        )
        total_duration_ms = self._extract_numeric_value(stats_payload, ("total_duration_ms",))
        total_duration_seconds = (total_duration_ms / 1000.0) if total_duration_ms is not None else wall_duration_seconds

        generation_seconds: float | None = None
        if completion_tokens_value is not None and tokens_per_second and tokens_per_second > 0:
            generation_seconds = completion_tokens_value / tokens_per_second
        elif ttft_seconds is not None and total_duration_seconds >= ttft_seconds:
            residual = total_duration_seconds - ttft_seconds
            if residual > 0:
                generation_seconds = residual

        prompt_tokens = int(prompt_tokens_value) if prompt_tokens_value is not None else None
        completion_tokens = int(completion_tokens_value) if completion_tokens_value is not None else None
        total_tokens_value = self._extract_numeric_value(stats_payload, ("total_tokens_count", "total_tokens"))
        total_tokens = int(total_tokens_value) if total_tokens_value is not None else None
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        stop_reason = self._extract_string_value(stats_payload, ("stop_reason",))
        model_id = self._extract_string_value(model_info_payload, ("model_key", "id", "display_name")) or self.model_tag
        architecture = self._extract_string_value(model_info_payload, ("architecture",))
        gpu_layers_value = self._extract_numeric_value(stats_payload, ("num_gpu_layers",))
        gpu_layers = int(gpu_layers_value) if gpu_layers_value is not None else None

        return InferenceTelemetry(
            total_duration_seconds=total_duration_seconds,
            ttft_seconds=ttft_seconds,
            generation_duration_seconds=generation_seconds,
            prompt_tokens_per_second=None,
            tokens_per_second=tokens_per_second,
            reasoning_tokens=None,
            stop_reason=stop_reason,
            model_id=model_id,
            vram_usage_mb=None,
            total_params=None,
            architecture=architecture,
            quantization=None,
            gpu_layers=gpu_layers,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            stats=stats_payload,
            resources={},
        )

    def _encode_image_data_url(self, image_path: str) -> str:
        """
        Codifica una imagen local en formato Data URL (base64).
        
        Args:
            image_path (str): Ruta al archivo de imagen.
            
        Returns:
            str: La cadena Data URL representando la imagen.
        """
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def _prepare_image_source_for_lms(self, image_path: str) -> Any:
        """
        Prepara y optimiza una imagen para ser enviada a través de la API de LM Studio.
        
        Redimensiona la imagen si es necesario para no exceder las dimensiones máximas
        soportadas y la convierte a formato PNG optimizado.
        
        Args:
            image_path (str): Ruta al archivo de imagen original.
            
        Returns:
            Any: Un objeto BytesIO con la imagen procesada o la ruta original en caso de error.
        """
        if Image is None:
            return image_path

        try:
            max_pixels = 1_800_000
            with Image.open(image_path) as image_obj:
                clean_image = image_obj.convert("RGB")
                width, height = clean_image.size
                current_pixels = width * height
                if current_pixels > max_pixels:
                    scale = math.sqrt(max_pixels / float(current_pixels))
                    resized_width = max(1, int(width * scale))
                    resized_height = max(1, int(height * scale))
                    resampling_namespace = getattr(Image, "Resampling", None)
                    resampling = getattr(resampling_namespace, "LANCZOS", getattr(Image, "LANCZOS"))
                    clean_image = clean_image.resize((resized_width, resized_height), resampling)

                buffer = io.BytesIO()
                clean_image.save(buffer, format="PNG", optimize=True)
                buffer.seek(0)

                base_name, _ = os.path.splitext(os.path.basename(image_path))
                buffer.name = f"{base_name}.png"
                return buffer
        except Exception:
            return image_path

    def _build_multimodal_history(self, structured_text: str, image_path: str) -> Any:
        """
        Construye el historial de chat multimodal compatible con el SDK de LM Studio.
        
        Maneja la preparación de la imagen y la creación del objeto Chat o estructura de mensaje
        adecuada según las capacidades del SDK disponible.
        
        Args:
            structured_text (str): El texto del prompt estructurado.
            image_path (str): Ruta a la imagen a analizar.
            
        Returns:
            Any: Un objeto Chat de LM Studio o un diccionario de historial de mensajes.
        """
        if self._client is not None:
            prepare_image_fn = getattr(getattr(self._client, "files", None), "prepare_image", None)
        else:
            prepare_image_fn = getattr(lms, "prepare_image", None) if lms is not None else None
        image_source = self._prepare_image_source_for_lms(image_path)
        if callable(prepare_image_fn):
            try:
                image_payload = prepare_image_fn(image_source)
            except Exception:
                try:
                    image_payload = prepare_image_fn(image_path)
                except Exception:
                    image_payload = self._encode_image_data_url(image_path)
        else:
            image_payload = self._encode_image_data_url(image_path)

        chat_cls = getattr(lms, "Chat", None) if lms is not None else None
        if callable(chat_cls):
            chat = cast(_LMSChatHandle, chat_cls())
            try:
                chat.add_user_message(structured_text, images=[image_payload])
                return chat
            except Exception:
                chat = cast(_LMSChatHandle, chat_cls())
                chat.add_user_message([structured_text, image_payload])
                return chat

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [structured_text, image_payload],
                }
            ]
        }

    def _respond_with_config_compat(
        self,
        model_handle: _LMSModelHandle,
        payload: Any,
        temperature: float,
        schema: type[BaseModel] | None = None,
    ) -> Any:
        """
        Ejecuta la inferencia intentando usar la configuración más estricta posible, con fallback.

        Intenta usar ``response_format`` para forzar la salida al esquema Pydantic indicado.
        Si el SDK no soporta algún argumento, reintenta con una configuración más simple.

        Args:
            model_handle: El manejador del modelo cargado.
            payload: El historial de chat o prompt.
            temperature (float): Temperatura para la generación.
            schema (type[BaseModel] | None): Clase Pydantic a usar como ``response_format``.
                Si es None se utiliza ``GenericObjectDetection``.
        Returns:
            Any: La respuesta cruda del modelo.
        """

        def unsupported_kwarg(error: TypeError, keyword: str) -> bool:
            """
            Detecta si un ``TypeError`` proviene de un argumento no soportado.

            Args:
                error (TypeError): Excepción capturada durante la llamada al SDK.
                keyword (str): Nombre del argumento que se quiere comprobar.

            Returns:
                bool: ``True`` si el error indica que ``keyword`` no está
                soportado por la versión actual del SDK.
            """

            error_text = str(error)
            return "unexpected keyword argument" in error_text and keyword in error_text

        # Usa el esquema inyectado o cae al contrato legado por defecto
        target_schema = schema if schema is not None else GenericObjectDetection

        base_config: dict[str, Any] = {"temperature": temperature}

        kwargs: dict[str, Any] = {
            "response_format": target_schema,
            "config": base_config,
        }

        attempts = 0
        while True:
            attempts += 1
            try:
                return model_handle.respond(payload, **kwargs)
            except TypeError as error:
                if attempts >= 6:
                    raise

                removed = False
                if "response_format" in kwargs and unsupported_kwarg(error, "response_format"):
                    kwargs.pop("response_format", None)
                    removed = True

                if "config" in kwargs and unsupported_kwarg(error, "config"):
                    kwargs.pop("config", None)
                    kwargs["temperature"] = temperature
                    removed = True

                if "temperature" in kwargs and unsupported_kwarg(error, "temperature"):
                    kwargs.pop("temperature", None)
                    removed = True

                if not removed:
                    raise

    def load_model(self, n_ctx: int = 8192, n_gpu_layers: int = -1) -> None:
        """
        Carga el modelo especificado en la configuración.
        
        Si ya hay un modelo cargado, no hace nada. Si el tag detectado es genérico ('auto', etc.),
        intenta usar el primer modelo disponible.
        
        Args:
            n_ctx (int, optional): Ventana de contexto en tokens. Se pasa a LM Studio como
                ``contextLength``. Por defecto 8192, apropiado para modelos VLM multimodales.
                Si es 0 o negativo, no se especifica ventana (LM Studio usa su valor por defecto).
            n_gpu_layers (int, optional): Capas en GPU (gestionado por LM Studio). Por defecto -1.
            
        Raises:
            RuntimeError: Si falla la inicialización del cliente o la carga del modelo.
        """
        _ = n_gpu_layers

        load_config: dict[str, Any] = {}
        if n_ctx and n_ctx > 0:
            load_config["contextLength"] = n_ctx

        self._ensure_lms_available()
        assert lms is not None

        if self._loaded_model is not None:
            return

        loaded_models = self._list_loaded_models_keys()
        if self.model_tag.lower() in {"auto", "local-model", ""} and loaded_models:
            self.model_tag = loaded_models[0]

        def _try_load(fn: Any) -> Any:
            """
            Intenta cargar el modelo con configuración explícita y fallback simple.

            Args:
                fn (Any): Función o callable del SDK capaz de cargar un modelo.

            Returns:
                Any: Manejador del modelo cargado por LM Studio.

            Raises:
                Exception: Propaga cualquier error distinto del ``TypeError`` de
                compatibilidad que activa el fallback sin configuración.
            """

            if load_config:
                try:
                    return fn(self.model_tag, config=load_config)
                except TypeError:
                    pass
            return fn(self.model_tag)

        try:
            self._client = self._create_lms_client()
            client = self._client
            if client is None:
                raise RuntimeError("No se pudo inicializar cliente LM Studio")
            model = _try_load(client.llm.model)
        except Exception as client_error:
            self._client = None
            try:
                model = _try_load(lms.llm)
            except Exception as primary_error:
                load_fn = getattr(getattr(lms, "llm", None), "load", None)
                if callable(load_fn):
                    try:
                        model = _try_load(load_fn)
                    except Exception as fallback_error:
                        for candidate_error in (client_error, primary_error, fallback_error):
                            if self._is_lmstudio_connection_issue(candidate_error):
                                raise RuntimeError(
                                    self._build_lmstudio_connection_message(
                                        f"cargar el modelo '{self.model_tag}'",
                                        candidate_error,
                                    )
                                ) from fallback_error
                        raise RuntimeError(
                            f"No se pudo cargar el modelo '{self.model_tag}' con LM Studio. "
                            f"Error client: {client_error}. "
                            f"Error principal: {primary_error}. Error fallback: {fallback_error}"
                        ) from fallback_error
                else:
                    for candidate_error in (client_error, primary_error):
                        if self._is_lmstudio_connection_issue(candidate_error):
                            raise RuntimeError(
                                self._build_lmstudio_connection_message(
                                    f"cargar el modelo '{self.model_tag}'",
                                    candidate_error,
                                )
                            ) from primary_error
                    raise RuntimeError(
                        f"No se pudo cargar el modelo '{self.model_tag}' con LM Studio. "
                        f"Error client: {client_error}. Error: {primary_error}"
                    ) from primary_error

        self._loaded_model = cast(_LMSModelHandle, model)

    def preload_model(self, keep_alive: str = "20m") -> None:
        """
        Precarga el modelo en memoria enviando una consulta mínima (warmup).
        
        Útil para asegurar que el modelo está listo antes de recibir tráfico real,
        reduciendo la latencia de la primera petición.
        
        Args:
            keep_alive (str, optional): Duración para mantener el modelo cargado.
                Por defecto es "20m" (20 minutos), aunque actualmente no se usa explícitamente.
                
        Raises:
            RuntimeError: Si falla la petición de calentamiento.
        """
        _ = keep_alive
        self.load_model()
        try:
            model_handle = self._loaded_model
            assert model_handle is not None
            model_handle.respond("Warmup mínimo. Responde solo: ok.")
        except Exception as error:
            if self._is_lmstudio_connection_issue(error):
                raise RuntimeError(
                    self._build_lmstudio_connection_message(
                        f"precargar el modelo '{self.model_tag}'",
                        error,
                    )
                ) from error
            raise RuntimeError(
                f"No se pudo precargar el modelo '{self.model_tag}' en LM Studio: {error}"
            ) from error

    def unload_model(self) -> None:
        """
        Descarga el modelo de la memoria y libera recurssos del cliente.
        
        Intenta cerrar limpiamente la conexión con LM Studio y liberar el modelo.
        No lanza excepciones si falla, solo limpia las referencias locales.
        """
        try:
            self._ensure_lms_available()
            if self._client is not None:
                unload_fn = getattr(getattr(self._client, "llm", None), "unload", None)
                if callable(unload_fn):
                    unload_fn(self.model_tag)
                close_fn = getattr(self._client, "close", None)
                if callable(close_fn):
                    close_fn()
                self._client = None
            else:
                unload_fn = getattr(getattr(lms, "llm", None), "unload", None)
                if callable(unload_fn):
                    unload_fn(self.model_tag)
        except Exception:
            pass
        finally:
            self._client = None
            self._loaded_model = None


    @overload
    def inference(
        self,
        image_path: str | Path,
        prompt: str,
        schema: Type[T] = GenericObjectDetection,
        temperature: float = 0.7,
        *,
        include_telemetry: Literal[False] = False,
    ) -> T:
        ...

    @overload
    def inference(
        self,
        image_path: str | Path,
        prompt: str,
        schema: Type[T] = GenericObjectDetection,
        temperature: float = 0.7,
        *,
        include_telemetry: Literal[True],
    ) -> InferenceResult[T]:
        ...

    def inference(
        self,
        image_path: str | Path,
        prompt: str,
        schema: Type[T] = GenericObjectDetection,  # type: ignore[assignment]
        temperature: float = 0.7,
        *,
        include_telemetry: bool = False,
    ) -> T | InferenceResult[T]:
        """
        Ejecuta una inferencia multimodal (VLM) sobre una imagen con esquema dinámico.

        Carga el modelo si es necesario, prepara la imagen y el prompt, y devuelve
        la respuesta *ya parseada y validada* al objeto Pydantic especificado por ``schema``.

        Uso básico::

            from src.inference.schemas import PolypDetection, SycophancyTest
            loader = VLMLoader("mi-modelo")

            # Detección básica
            result: PolypDetection = loader.inference("imagen.jpg", prompt, schema=PolypDetection)

        Args:
            image_path (str | Path): Ruta al archivo de imagen (local).
            prompt (str): Pregunta o instrucción para el modelo.
            schema (Type[T]): Clase Pydantic que define el contrato de respuesta JSON.
                Por defecto usa ``GenericObjectDetection`` para compatibilidad hacia atrás.
            temperature (float, optional): Creatividad de la respuesta (0.0 a 2.0). Default 0.7.
            include_telemetry (bool, optional): Si es True, devuelve además métricas
                de rendimiento y uso del SDK junto al resultado parseado.

        Returns:
            T | InferenceResult[T]: Instancia validada del esquema, o un contenedor
            con el dato validado y la telemetría si ``include_telemetry`` es True.

        Raises:
            ValueError: Si path o prompt están vacíos, o temperatura fuera de rango.
            FileNotFoundError: Si la imagen no existe en el sistema de archivos.
            RuntimeError: Si hay fallos de conexión con LM Studio, de inferencia,
                procesamiento de imagen o validación del esquema.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt no puede estar vacío")

        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as error:
            raise ValueError("temperature debe ser numérico entre 0.0 y 2.0") from error

        if not (0.0 <= temperature_value <= 2.0):
            raise ValueError("temperature debe estar entre 0.0 y 2.0")

        # Normaliza la ruta: acepta str o Path
        image_path_str = str(image_path).strip()
        if not image_path_str:
            raise ValueError("image_path no puede estar vacío")

        if not os.path.isfile(image_path_str):
            raise FileNotFoundError(f"Imagen no encontrada en: {image_path_str}")

        # Carga el modelo si aún no está en memoria
        self.load_model()
        assert self._loaded_model is not None

        # Genera el prompt con instrucciones de salida JSON basadas en el esquema
        structured_text = self._build_structured_instruction(prompt, schema=schema)

        multimodal_input = self._build_multimodal_history(structured_text, image_path_str)

        model_handle = self._loaded_model
        assert model_handle is not None

        started_at = time.perf_counter()
        try:
            response = self._respond_with_config_compat(
                model_handle,
                multimodal_input,
                temperature_value,
                schema=schema,
            )
        except Exception as image_error:
            if self._is_lmstudio_connection_issue(image_error):
                raise RuntimeError(
                    self._build_lmstudio_connection_message("ejecutar la inferencia", image_error)
                ) from image_error
            try:
                self._respond_with_config_compat(
                    model_handle,
                    structured_text,
                    temperature_value,
                    schema=schema,
                )
            except Exception as fallback_error:
                if self._is_lmstudio_connection_issue(fallback_error):
                    raise RuntimeError(
                        self._build_lmstudio_connection_message(
                            "ejecutar la inferencia",
                            fallback_error,
                        )
                    ) from image_error
                raise RuntimeError(
                    "Fallo de inferencia en LM Studio: no se pudo procesar la imagen "
                    f"ni ejecutar fallback de texto. Detalle: {fallback_error}"
                ) from image_error

            raise RuntimeError(
                "Fallo de inferencia multimodal en LM Studio: error al manejar la imagen. "
                "Se verificó fallback de texto, pero esta operación requiere imagen válida."
            ) from image_error

        total_duration_seconds = time.perf_counter() - started_at
        telemetry = self._extract_inference_telemetry(response, total_duration_seconds)

        # Intenta usar .parsed antes de parsear el texto a mano
        parsed_payload = getattr(response, "parsed", None)
        if isinstance(parsed_payload, dict):
            try:
                validated = schema.model_validate(parsed_payload)
                telemetry = dataclasses.replace(
                    telemetry,
                    reasoning_tokens=self._estimate_reasoning_tokens(validated),
                )
                if include_telemetry:
                    return InferenceResult(data=validated, telemetry=telemetry)
                return validated  # type: ignore[return-value]
            except ValidationError as error:
                raise RuntimeError(
                    f"La respuesta estructurada no cumple el esquema '{schema.__name__}': {error}"
                ) from error

        response_text = self._extract_response_text(response)

        if not response_text:
            raise RuntimeError("La respuesta del modelo está vacía o no contiene texto útil.")

        try:
            parsed_json = json.loads(response_text)
            validated = schema.model_validate(parsed_json)
            telemetry = dataclasses.replace(
                telemetry,
                reasoning_tokens=self._estimate_reasoning_tokens(validated),
            )
            if include_telemetry:
                return InferenceResult(data=validated, telemetry=telemetry)
            return validated  # type: ignore[return-value]
        except json.JSONDecodeError as error:
            raise RuntimeError(f"La respuesta no es JSON válido: {error}") from error
        except ValidationError as error:
            raise RuntimeError(
                f"La respuesta no cumple el esquema '{schema.__name__}': {error}"
            ) from error

