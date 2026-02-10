# TFG: Generació de Descripcions Explicatives en Imatges Mèdiques amb VLMs

**Estudiant:** David Bonilla Medina  
**Grau:** Enginyeria Informàtica (Menció Computació) - UAB  
**Curs:** 2025/2026

## 📋 Descripció
Aquest projecte explora l'ús de **Models de Llenguatge Visual (VLMs)** d'última generació (SOTA 2026) i arquitectura *Open Source* per generar descripcions clíniques detallades (*Explainability*) d'imatges de colonoscòpia (pòlips).

S'analitzaran i compararan models com **Qwen3-VL**, **MiniCPM-o 4.5** i **InternVL 3.5**, centrant-se en l'ús de noves tecnologies d'encoder visual (**SigLIP 2**) i mecanismes de resolució dinàmica.

L'objectiu és demostrar la viabilitat d'executar aquests sistemes en **entorns locals** (Edge AI) utilitzant maquinari de consum (RTX 4060 Ti), garantint la privacitat de les dades mèdiques.

## 🛠️ Stack Tecnològic
*   **Llenguatge:** Python 3.12
*   **Gestor de Paquets:** `uv` (Astral)
*   **Frameworks:** PyTorch (CUDA 12.1), Hugging Face Transformers
*   **Inferencia Local:** `llama-cpp-python` (GGUF), `bitsandbytes`

## 🚀 Instal·lació i Configuració

Aquest projecte utilitza **uv** per a una gestió ràpida de dependències.

1.  **Clonar el repositori:**
    ```bash
    git clone https://github.com/NIU1635219/TFG_VLM_Medical.git
    cd TFG_VLM_Medical
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
