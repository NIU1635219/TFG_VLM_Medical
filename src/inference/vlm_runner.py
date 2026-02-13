"""Módulo de carga e inferencia VLM sobre Ollama."""

import os

try:
    import ollama
except ImportError:
    ollama = None

class VLMLoader:
    """
    Clase encargada de gestionar la interacción con Ollama para modelos VLM.
    Mantiene la interfaz original para compatibilidad con scripts existentes.
    """

    def __init__(self, model_path: str, verbose: bool = False):
        """
        Inicializa la configuración del cargador.

        Args:
            model_path (str): Nombre del tag del modelo en Ollama (ej. 'minicpm-v:latest').
                              Se mantiene el nombre 'model_path' por compatibilidad.
            verbose (bool): Si es True, imprime logs detallados.
        """
        if not model_path or not str(model_path).strip():
            raise ValueError("model_path/model_tag no puede estar vacío")

        self.model_tag = str(model_path).strip()
        self.verbose = verbose

        self._ensure_ollama_available()

    def _ensure_ollama_available(self) -> None:
        if ollama is None:
            raise RuntimeError("La librería 'ollama' no está instalada. Ejecuta: uv pip install ollama")

    def _extract_available_tags(self, models_response) -> list[str]:
        """Normaliza la salida de `ollama.list()` para extraer tags de modelo."""
        models = None

        if hasattr(models_response, "models"):
            models = models_response.models
        elif isinstance(models_response, dict):
            models = models_response.get("models", [])

        if not models:
            return []

        tags: list[str] = []
        for model_item in models:
            model_tag = getattr(model_item, "model", None)
            if model_tag is None and isinstance(model_item, dict):
                model_tag = model_item.get("model") or model_item.get("name")

            if isinstance(model_tag, str) and model_tag.strip():
                tags.append(model_tag.strip())

        return tags

    def _extract_chat_content(self, response) -> str:
        """Extrae texto de respuesta para distintos formatos del cliente Ollama."""
        # Caso dict: {'message': {'content': '...'}}
        if isinstance(response, dict):
            message = response.get("message", {})
            if isinstance(message, dict):
                return str(message.get("content", "") or "")

        # Caso objeto: response.message.content
        message_obj = getattr(response, "message", None)
        if message_obj is not None:
            # message puede ser dict u objeto
            if isinstance(message_obj, dict):
                return str(message_obj.get("content", "") or "")
            content_attr = getattr(message_obj, "content", None)
            if content_attr is not None:
                return str(content_attr)

        # Fallback extremadamente defensivo
        content_attr = getattr(response, "content", None)
        if content_attr is not None:
            return str(content_attr)

        return ""

    def load_model(self, n_ctx: int = 2048, n_gpu_layers: int = -1) -> None:
        """
        Verifica si el modelo está disponible en Ollama. 
        En Ollama no se 'carga' explícitamente en una variable, el servidor lo gestiona.
        
        Args:
            n_ctx (int): Ignorado en Ollama (configurado en el servidor/modelfile).
            n_gpu_layers (int): Ignorado en Ollama (gestionado automáticamente).
        """
        _ = n_ctx
        _ = n_gpu_layers

        if self.verbose:
            print(f"ℹ️ Verificando disponibilidad del modelo: {self.model_tag} en Ollama...")

        self._ensure_ollama_available()

        try:
            assert ollama is not None
            models_response = ollama.list()
            available_tags = self._extract_available_tags(models_response)
            
            if self.model_tag not in available_tags:
                print(f"⚠️ El modelo {self.model_tag} no aparece en 'ollama list'. Intentando pull automático...")
                print(f"⏳ Pulling {self.model_tag} (esto puede tardar)...")

                for progress in ollama.pull(self.model_tag, stream=True):
                    if self.verbose and isinstance(progress, dict) and progress.get("status"):
                        print(f"\r Status: {progress['status']}", end="", flush=True)
                if self.verbose:
                    print("\n✅ Pull completado.")
            elif self.verbose:
                print(f"✅ Modelo {self.model_tag} detectado y listo.")

        except Exception as e:
            raise RuntimeError(
                f"Error conectando con Ollama: {str(e)}. "
                "Asegúrate de que 'ollama serve' está ejecutándose."
            ) from e

    def preload_model(self, keep_alive: str = "20m") -> None:
        """
        Precarga el modelo en memoria para reutilizarlo durante múltiples inferencias.

        Args:
            keep_alive (str): Ventana de permanencia en memoria (formato Ollama, ej. "20m").
        """
        self.load_model()

        if self.verbose:
            print(f"⏳ Precargando modelo {self.model_tag} (keep_alive={keep_alive})...")

        try:
            assert ollama is not None
            ollama.chat(
                model=self.model_tag,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Warmup mínimo. Responde solo: ok.'
                    }
                ],
                options={
                    'temperature': 0
                },
                keep_alive=keep_alive,
            )
            if self.verbose:
                print("✅ Modelo precargado.")
        except Exception as e:
            raise RuntimeError(f"No se pudo precargar el modelo {self.model_tag}: {str(e)}") from e

    def unload_model(self) -> None:
        """
        Solicita a Ollama liberar el modelo de memoria.
        """
        if self.verbose:
            print(f"🧹 Liberando modelo {self.model_tag} de memoria...")

        try:
            assert ollama is not None
            ollama.chat(
                model=self.model_tag,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Finalizar sesión.'
                    }
                ],
                options={
                    'temperature': 0
                },
                keep_alive=0,
            )
            if self.verbose:
                print("✅ Modelo liberado.")
        except Exception as e:
            raise RuntimeError(f"No se pudo liberar el modelo {self.model_tag}: {str(e)}") from e


    def inference(self, image_path: str, prompt: str, temperature: float = 0.7) -> str:
        """
        Realiza la inferencia sobre una imagen dada un prompt de texto.

        Args:
            image_path (str): Ruta al archivo de imagen.
            prompt (str): Instrucción para el modelo.
            temperature (float): Creatividad del modelo.

        Returns:
            str: Respuesta generada por el modelo.
        """
        self._ensure_ollama_available()

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt no puede estar vacío")

        if not (0.0 <= float(temperature) <= 2.0):
            raise ValueError("temperature debe estar entre 0.0 y 2.0")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada en: {image_path}")

        if self.verbose:
            print(f"🚀 Enviando petición a Ollama [{self.model_tag}]...")

        try:
            assert ollama is not None
            response = ollama.chat(
                model=self.model_tag,
                messages=[
                    {
                        'role': 'system', 
                        'content': 'Responde siempre en español.'
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                ],
                options={
                    'temperature': temperature
                }
            )
            return self._extract_chat_content(response)

        except Exception as e:
            raise RuntimeError(f"Error durante la inferencia con Ollama: {str(e)}") from e

