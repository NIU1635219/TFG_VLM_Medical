"""
Módulo para la carga e inferencia de modelos Vision-Language (VLM) usando llama-cpp-python.
Diseñado específicamente para MiniCPM-V y otros modelos compatibles con formato GGUF.
"""

from typing import Optional, Dict, Any, List
import os

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
except ImportError:
    # Graceful fallback for environments where llama-cpp-python isn't installed (e.g. CI without GPU)
    Llama = None
    MiniCPMv26ChatHandler = None

class VLMLoader:
    """
    Clase encargada de gestionar el ciclo de vida del modelo VLM:
    configuración, carga en memoria (GPU/CPU) e inferencia.
    """

    def __init__(self, model_path: str, verbose: bool = False):
        """
        Inicializa la configuración del cargador, pero NO carga el modelo todavía.

        Args:
            model_path (str): Ruta absoluta o relativa al archivo .gguf del modelo.
            verbose (bool): Si es True, imprime logs detallados de llama.cpp.

        Raises:
            FileNotFoundError: Si el archivo model_path no existe.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"VLM Model not found at: {model_path}")
        
        self.model_path = model_path
        self.verbose = verbose
        self.llm: Optional[Any] = None

    def load_model(self, n_ctx: int = 2048, n_gpu_layers: int = -1) -> None:
        """
        Carga el modelo en memoria.

        Args:
            n_ctx (int): Tamaño del contexto (tokens de texto + parches de imagen).
            n_gpu_layers (int): Número de capas a mover a la GPU. -1 para todas.

        Raises:
            RuntimeError: Si falla la carga del modelo (ej. falta de memoria, formato inválido).
        """
        if Llama is None or MiniCPMv26ChatHandler is None:
            raise RuntimeError("llama-cpp-python no está instalado correctamente.")

        try:
            # Autodetectar el proyector (mmproj) en la misma carpeta que el modelo
            model_dir = os.path.dirname(self.model_path)
            # Buscamos archivos que contengan 'mmproj' en la misma carpeta
            mmproj_files = [f for f in os.listdir(model_dir) if "mmproj" in f and f.endswith(".gguf")]
            
            if mmproj_files:
                mmproj_path = os.path.join(model_dir, mmproj_files[0])
                print(f"ℹ️ Cargando proyector de visión detectado: {mmproj_path}")
            else:
                print(f"⚠️ Advertencia: No se encontró un archivo mmproj en {model_dir}, usando el modelo principal como fallback.")
                mmproj_path = self.model_path

            # Nota: MiniCPMv26ChatHandler es compatible con la arquitectura de la v4.5 en llama-cpp-python
            chat_handler = MiniCPMv26ChatHandler(clip_model_path=mmproj_path, verbose=self.verbose)
            
            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=chat_handler,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=self.verbose,
            )
        except Exception as e:
            raise RuntimeError(f"Error cargando el modelo: {str(e)}")

    def inference(self, image_path: str, prompt: str, temperature: float = 0.7) -> str:
        """
        Realiza la inferencia sobre una imagen dada un prompt de texto.

        Args:
            image_path (str): Ruta al archivo de imagen.
            prompt (str): Instrucción para el modelo (ej. "Describe esta imagen").
            temperature (float): Creatividad del modelo (0.0 - 1.0).

        Returns:
            str: Respuesta generada por el modelo.

        Raises:
            RuntimeError: Si el modelo no ha sido cargado previamente.
            FileNotFoundError: Si la imagen no existe.
        """
        if self.llm is None:
            raise RuntimeError("Modelo no cargado. Ejecuta load_model() primero.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada en: {image_path}")

        # Construir el mensaje formato OpenAI Vision API para llama-cpp
        # Nota: Usamos file:// URL normalizado
        import pathlib
        abs_image_path = os.path.abspath(image_path)
        # Convertir a URI compatible con file:/// (maneja backslashes y espacios)
        file_uri = pathlib.Path(abs_image_path).as_uri()
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": file_uri}}
                ]
            }
        ]

        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=1024 # Límite razonable para descripciones médicas
            )
            
            # Extraer contenido de la respuesta
            if response and "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["message"]["content"]
            else:
                return ""
        except Exception as e:
            raise RuntimeError(f"Error durante la inferencia: {str(e)}")
