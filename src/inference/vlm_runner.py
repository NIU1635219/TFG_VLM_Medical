"""Módulo de carga e inferencia VLM sobre LM Studio usando lmstudio-python."""

import base64
import io
import json
import mimetypes
import os
from typing import Any, Protocol, cast

from pydantic import BaseModel, ValidationError

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import lmstudio as lms
except ImportError:
    lms = None


class VLMStructuredResponse(BaseModel):
    """Contrato estructurado obligatorio para inferencia clínica."""

    polyp_detected: bool
    confidence_score: int
    justification: str


class _LMSModelHandle(Protocol):
    def respond(self, history: Any, **kwargs: Any) -> Any:
        ...


class _LMSChatHandle(Protocol):
    def add_user_message(self, content: Any, *, images: Any = (), _files: Any = ()) -> Any:
        ...


class VLMLoader:
    """Gestor de carga e inferencia para modelos VLM en LM Studio."""

    def __init__(
        self,
        model_path: str,
        verbose: bool = False,
        server_api_host: str | None = None,
        api_token: str | None = None,
    ):
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
        """Construye cliente scoped de LM Studio con host/token opcionales."""
        assert lms is not None
        client_cls = getattr(lms, "Client", None)
        if not callable(client_cls):
            raise RuntimeError("Client API no disponible")

        kwargs: dict[str, Any] = {}
        if self.api_token:
            kwargs["api_token"] = self.api_token

        if self.server_api_host:
            return client_cls(self.server_api_host, **kwargs)
        return client_cls(**kwargs)

    def _ensure_lms_available(self) -> None:
        """Verifica que lmstudio-python esté disponible."""
        if lms is None:
            raise RuntimeError(
                "La librería 'lmstudio-python' no está instalada. "
                "Instala y activa LM Studio SDK para continuar."
            )

    def _extract_model_key(self, model_obj: Any) -> str | None:
        """Extrae una clave de modelo desde dict u objeto."""
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
        """Lista claves de modelos cargados; devuelve [] si falla."""
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

    def _build_structured_instruction(self, prompt: str) -> str:
        """Construye instrucción estricta para salida JSON."""
        prompt_clean = prompt.strip()
        return (
            f"{prompt_clean}\n\n"
            "Responde EXCLUSIVAMENTE con JSON válido, sin texto adicional, con estas claves exactas:\n"
            "- polyp_detected (boolean)\n"
            "- confidence_score (integer 0-100)\n"
            "- justification (string)\n"
            "No incluyas markdown, comentarios ni explicaciones fuera del JSON."
        )

    def _extract_response_text(self, response: Any) -> str:
        """Extrae contenido textual desde distintas formas de respuesta."""

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

    def _encode_image_data_url(self, image_path: str) -> str:
        """Convierte imagen local a data URL base64."""
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{encoded}"

    def _prepare_image_source_for_lms(self, image_path: str) -> Any:
        """Normaliza imagen para LM Studio y devuelve source compatible con prepare_image."""
        if Image is None:
            return image_path

        try:
            max_dim = 1024
            with Image.open(image_path) as image_obj:
                clean_image = image_obj.convert("RGB")
                if max(clean_image.size) > max_dim:
                    clean_image.thumbnail((max_dim, max_dim))

                buffer = io.BytesIO()
                clean_image.save(buffer, format="PNG", optimize=True)
                buffer.seek(0)

                base_name, _ = os.path.splitext(os.path.basename(image_path))
                buffer.name = f"{base_name}.png"
                return buffer
        except Exception:
            return image_path

    def _build_multimodal_history(self, structured_text: str, image_path: str) -> Any:
        """Construye historial multimodal compatible con LM Studio SDK."""
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

    def _respond_with_config_compat(self, model_handle: _LMSModelHandle, payload: Any, temperature: float) -> Any:
        """Llama a respond priorizando response_format/config con degradación compatible."""

        def unsupported_kwarg(error: TypeError, keyword: str) -> bool:
            error_text = str(error)
            return "unexpected keyword argument" in error_text and keyword in error_text

        kwargs: dict[str, Any] = {
            "response_format": VLMStructuredResponse,
            "config": {"temperature": temperature},
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

    def load_model(self, n_ctx: int = 2048, n_gpu_layers: int = -1) -> None:
        _ = n_ctx
        _ = n_gpu_layers

        self._ensure_lms_available()
        assert lms is not None

        if self._loaded_model is not None:
            return

        loaded_models = self._list_loaded_models_keys()
        if self.model_tag.lower() in {"auto", "local-model", ""} and loaded_models:
            self.model_tag = loaded_models[0]

        try:
            self._client = self._create_lms_client()
            client = self._client
            if client is None:
                raise RuntimeError("No se pudo inicializar cliente LM Studio")
            model = client.llm.model(self.model_tag)
        except Exception as client_error:
            self._client = None
            try:
                model = lms.llm(self.model_tag)
            except Exception as primary_error:
                load_fn = getattr(getattr(lms, "llm", None), "load", None)
                if callable(load_fn):
                    try:
                        model = load_fn(self.model_tag)
                    except Exception as fallback_error:
                        raise RuntimeError(
                            f"No se pudo cargar el modelo '{self.model_tag}' con LM Studio. "
                            f"Error client: {client_error}. "
                            f"Error principal: {primary_error}. Error fallback: {fallback_error}"
                        ) from fallback_error
                else:
                    raise RuntimeError(
                        f"No se pudo cargar el modelo '{self.model_tag}' con LM Studio. "
                        f"Error client: {client_error}. Error: {primary_error}"
                    ) from primary_error

        self._loaded_model = cast(_LMSModelHandle, model)

    def preload_model(self, keep_alive: str = "20m") -> None:
        _ = keep_alive
        self.load_model()
        try:
            model_handle = self._loaded_model
            assert model_handle is not None
            model_handle.respond("Warmup mínimo. Responde solo: ok.")
        except Exception as error:
            raise RuntimeError(
                f"No se pudo precargar el modelo '{self.model_tag}' en LM Studio: {error}"
            ) from error

    def unload_model(self) -> None:
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


    def inference(self, image_path: str, prompt: str, temperature: float = 0.7) -> VLMStructuredResponse:
        """Ejecuta inferencia VLM y devuelve objeto tipado validado."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt no puede estar vacío")

        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as error:
            raise ValueError("temperature debe ser numérico entre 0.0 y 2.0") from error

        if not (0.0 <= temperature_value <= 2.0):
            raise ValueError("temperature debe estar entre 0.0 y 2.0")

        if not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("image_path no puede estar vacío")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Imagen no encontrada en: {image_path}")

        self.load_model()
        assert self._loaded_model is not None

        structured_text = self._build_structured_instruction(prompt)

        multimodal_input = self._build_multimodal_history(structured_text, image_path)

        model_handle = self._loaded_model
        assert model_handle is not None

        try:
            response = self._respond_with_config_compat(
                model_handle,
                multimodal_input,
                temperature_value,
            )
        except Exception as image_error:
            try:
                self._respond_with_config_compat(
                    model_handle,
                    structured_text,
                    temperature_value,
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    "Fallo de inferencia en LM Studio: no se pudo procesar la imagen "
                    f"ni ejecutar fallback de texto. Detalle: {fallback_error}"
                ) from image_error

            raise RuntimeError(
                "Fallo de inferencia multimodal en LM Studio: error al manejar la imagen. "
                "Se verificó fallback de texto, pero esta operación requiere imagen válida."
            ) from image_error

        parsed_payload = getattr(response, "parsed", None)
        if isinstance(parsed_payload, dict):
            try:
                return VLMStructuredResponse.model_validate(parsed_payload)
            except ValidationError as error:
                raise RuntimeError(
                    f"La respuesta estructurada no cumple el esquema esperado: {error}"
                ) from error

        response_text = self._extract_response_text(response)

        if not response_text:
            raise RuntimeError("La respuesta del modelo está vacía o no contiene texto útil.")

        try:
            parsed_json = json.loads(response_text)
            return VLMStructuredResponse.model_validate(parsed_json)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"La respuesta no es JSON válido: {error}") from error
        except ValidationError as error:
            raise RuntimeError(
                f"La respuesta no cumple el esquema esperado: {error}"
            ) from error

