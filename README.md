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

## 🛠️ Manager Tool (v4.0) - CLI Interactive
El projecte inclou una potent eina de gestió (`setup_env.py`) amb una **Interfície d'Usuari de Text (TUI)** avançada que facilita la configuració, el diagnòstic i el manteniment de l'entorn.

**Funcionalitats Principals:**
*   **🎮 Interfície Gràfica en Terminal:** Navegació intuïtiva, circular i sense parpadeigs (*Flicker-Free*), amb indicadors de desplaçament (*Scrolling Wrappers*).
*   **🩺 Diagnòstic Dinàmic:** Analitza l'estat de 16+ llibreries crítiques i la configuració de CUDA. La taula es refresca automàticament després de cada solució aplicada.
*   **🤖 Gestió de Models VLM:** Sistema organitzat de models en subcarpetes per evitar conflictes entre proyector visuals (`mmproj`). Detecta i descarrega automàticament els fitxers necessaris segons la versió (MiniCPM-V 2.6, 4.5, etc.).
*   **🧪 Smoke Test Interactiu:** Prova d'inferència completa amb selecció de model i càrrega dinàmica en VRAM (GPU/CPU).
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

## � Compilació Avançada (Bleeding Edge)
Per a usuaris que necessitin suport per a models molt recents (com **MiniCPM-V 4.5** o **Qwen2.5-VL**) o vulguin maximitzar el rendiment amb CUDA utilitzant les últimes millores del backend C++:

El projecte inclou una eina especialitzada `setup_llama_cpp.py` que automatitza la compilació "Frankenstein" (Python wrapper estable + C++ backend `master`).

**Execució:**
```bash
python setup_llama_cpp.py
```

**Funcionalitats de l'Eina:**
1.  **🚀 Instal·lació Completa:** Descarrega `llama.cpp` (submòdul) directament de la branca `master`, el compila amb CUDA i instal·la el wrapper.
2.  **🩹 Binding Repair:** Detecta i corregeix automàticament les incompatibilitats entre el wrapper de Python i les funcions C++ eliminades en versions recents d'upstream.
3.  **📂 Detecció Intel·ligent:** Busca carpetes de codi font existents per evitar descàrregues innecessàries.
4.  **🧹 Neteja:** Gestiona els conflictes de bloqueig d'arxius a Windows per assegurar una compilació neta.

## �📂 Estructura del Projecte
L'arquitectura del projecte està dissenyada per ser modular i suportar múltiples architectures VLM:
*   `src/inference/`: Controladors d'inferència optimitzats per VLMs (GGUF + mmproj).
*   `src/scripts/`: Utilitats de terminal (test d'inferència interactiva, etc).
*   `models/`: (Ignorat per git) Models organitzats en subcarpetes (`minicpm_v26/`, `minicpm_v45/`) per evitar col·lisions de predictors visuals.
*   `notebooks/`: Proves i experiments controlats (Jupyter).
*   `setup_env.py`: Script de gestió v4.0 (TUI). No editar manualment.
*   `data/`: Dataset mèdic segmentat en `raw/` i `processed/`.

## 🤖 Models VLM Suportats
Actualment, el sistema està optimitzat per a la família **MiniCPM-V** de OpenBMB:
| Model | Configuració | Versió | Optimització |
| :--- | :--- | :--- | :--- |
| **MiniCPM-V 2.6** | Multi-crop / HD | 2.6 (GGUF) | 3.5GB-6GB VRAM |
| **MiniCPM-o 4.5** | High Res / SOTA | 4.5 (GGUF) | 8GB+ VRAM |

*Nota: El gestor detecta automàticament el fitxer `mmproj` corresponent dins de cada carpeta de model.*
