# TFG: Generació de Descripcions Explicatives en Imatges Mèdiques amb VLMs

**Estudiant:** David Bonilla Medina  
**Grau:** Enginyeria Informàtica (Menció Computació) - UAB  
**Curs:** 2025/2026

## 📋 Descripció
Aquest projecte explora l'ús de **Models de Llenguatge Visual (VLMs)** d'arquitectura *Open Source* (com Qwen2.5-VL, MiniCPM, InternVL) per generar descripcions clíniques detallades (*Explainability*) d'imatges de colonoscòpia (pòlips).

L'objectiu és demostrar la viabilitat d'executar aquests sistemes en **entorns locals** (Edge AI) utilitzant maquinari de consum, garantint la privacitat de les dades mèdiques.

## 🛠️ Stack Tecnològic
*   **Llenguatge:** Python 3.12
*   **Gestor de Paquets:** `uv` (Astral)
*   **Frameworks:** PyTorch (CUDA 12.1), Hugging Face Transformers
*   **Inferencia Local:** `llama-cpp-python` (GGUF), `bitsandbytes`

## 🚀 Instal·lació i Configuració

Aquest projecte utilitza **uv** per a una gestió ràpida de dependències.

1.  **Clonar el repositori:**
    ```bash
    git clone <URL_DEL_TEU_REPO>
    cd TFG_VLM
    ```

2.  **Crear l'entorn virtual:**
    ```bash
    uv venv .venv --python 3.12
    # Windows:
    .venv\Scripts\activate
    ```

3.  **Instalar dependències (GPU NVIDIA):**
    ```bash
    # PyTorch amb CUDA 12.1
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Dependències base
    uv pip install transformers accelerate protobuf scipy requests tqdm opencv-python bitsandbytes
    
    # Llama-cpp amb acceleració GPU
    $env:CMAKE_ARGS = "-DGGML_CUDA=on"
    uv pip install llama-cpp-python
    ```

## 📂 Estructura del Projecte
*   `src/`: Codi font dels scripts de processament i inferència.
*   `notebooks/`: proves i experiments (Jupyter).
*   `models/`: (Ignorat per git) Carpeta per desar els fitxers .gguf.
*   `data/`: (Ignorat per git) Dataset d'imatges mèdiques.
