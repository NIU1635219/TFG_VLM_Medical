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

## 🛠️ Manager Tool (v3.7) - CLI Interactive
El projecte inclou una potent eina de gestió (`setup_env.py`) amb una **Interfície d'Usuari de Text (TUI)** avançada que facilita la configuració, le diagnòstic i el manteniment de l'entorn.

**Funcionalitats Principals:**
*   **🎮 Interfície Gràfica en Terminal:** Navegació intuïtiva, circular i sense parpadeigs (*Flicker-Free*), amb indicadors de desplaçament (*Scrolling Wrappers*).
*   **🩺 Diagnòstic Profund:** Analitza l'estat de 16+ llibreries crítiques (incloent `bitsandbytes`, `scipy`, `cv2`) i la configuració de CUDA.
*   **🛡️ Dependency Resolution:** Sistema de reparació que resol automàticament les dependències faltants durant la reinstal·lació.
*   **📂 Gestió Granular de Llibreries:** Submenús desplegables que permeten seleccionar i reinstal·lar llibreries individuals.
*   **🔄 Auto-Restart:** El sistema detecta canvis crítics (com Torch/CUDA) i es reinicia automàticament per aplicar-los netament.

**Controls:**
*   `⬆️` / `⬇️`: Navegar per les opcions (rotació intel·ligent per nivells).
*   `ESPAI`: Entrar en Submenú / Marcar o desmarcar opcions.
*   `ESC`: Tornar enrere (tancar submenú) o sortir.
*   `ENTER`: **Confirmar i Executar** la selecció actual.

## 🚀 Instal·lació i Configuració

1.  **Clonar el repositori:**
    ```bash
    git clone https://github.com/NIU1635219/TFG_VLM_Medical.git
    cd TFG_VLM_Medical
    ```

2.  **Executar el Manager Tool:**
    Simplement executa l'script d'inici. Aquest llançarà el Manager Tool per configurar tot l'entorn.
    ```bash
    # Windows
    .\setup.bat

    # Linux / Mac
    chmod +x setup.sh
    ./setup.sh
    ```

3.  **Primer Ús:**
    La primera vegada que s'executi, l'eina detectarà que no existeix un entorn virutal i l'instal·larà automàticament. Després, podràs accedir al menú principal per verificar la instal·lació usant l'opció **Run System Diagnostics**.

4.  **Activar entorn:**
    ```bash
    # Windows:
    .venv\Scripts\activate
    ```

## 📂 Estructura del Projecte
*   `src/`: Codi font dels scripts de processament i inferència.
*   `notebooks/`: Proves i experiments (Jupyter).
*   `setup_env.py`: Script principal de gestió de l'entorn (**No editar manualment**).
*   `models/`: (Ignorat per git) Carpeta per desar els fitxers .gguf.
*   `data/`: (Ignorat per git) Dataset d'imatges mèdiques.
