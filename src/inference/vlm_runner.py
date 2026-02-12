"""
Módulo para la carga e inferencia de modelos Vision-Language (VLM) usando Ollama.
Reemplaza la implementación anterior de llama-cpp-python.
"""

from typing import Optional, Dict, Any, List
import os
import sys

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
        self.model_tag = model_path
        self.verbose = verbose
        
        if ollama is None:
            raise RuntimeError("La librería 'ollama' no está instalada. Ejecuta: uv pip install ollama")

    def load_model(self, n_ctx: int = 2048, n_gpu_layers: int = -1) -> None:
        """
        Verifica si el modelo está disponible en Ollama. 
        En Ollama no se 'carga' explícitamente en una variable, el servidor lo gestiona.
        
        Args:
            n_ctx (int): Ignorado en Ollama (configurado en el servidor/modelfile).
            n_gpu_layers (int): Ignorado en Ollama (gestionado automáticamente).
        """
        if self.verbose:
            print(f"ℹ️ Verificando disponibilidad del modelo: {self.model_tag} en Ollama...")

        try:
            # Listar modelos disponibles
            models_response = ollama.list()
            # models_response['models'] es una lista de objetos, cada uno tiene attr 'model'
            available_tags = [m.model for m in models_response.models]
            
            # Verificar si nuestro tag está (o si es substring, por temas de :latest)
            # Ollama tags suelen ser 'model:tag'
            if self.model_tag not in available_tags:
                print(f"⚠️ El modelo {self.model_tag} no aparece en 'ollama list'. Intentando pull automático...")
                print(f"⏳ Pulling {self.model_tag} (esto puede tardar)...")
                # Stream pull progress
                current_digest = None
                for progress in ollama.pull(self.model_tag, stream=True):
                    # Simple visual feedback
                    if progress.get('status') and self.verbose:
                        print(f"\r Status: {progress['status']}", end="", flush=True)
                print("\n✅ Pull completado.")
            else:
                if self.verbose:
                    print(f"✅ Modelo {self.model_tag} detectado y listo.")

        except Exception as e:
            # Si falla la conexión, es probable que Ollama no esté corriendo
            raise RuntimeError(f"Error conectando con Ollama: {str(e)}. Asegúrate de que 'ollama serve' está ejecutándose.")


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
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada en: {image_path}")

        if self.verbose:
            print(f"🚀 Enviando petición a Ollama [{self.model_tag}]...")

        try:
            # Ollama python lib espera path de imagen como string o bytes
            # Automáticamente maneja base64 si se le pasa el path
            
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
            
            # La respuesta viene en response['message']['content']
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                return ""

        except Exception as e:
            raise RuntimeError(f"Error durante la inferencia con Ollama: {str(e)}")

