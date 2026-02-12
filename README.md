# TFG: Generació de Descripcions Explicatives en Imatges Mèdiques amb VLMs

**Estudiant:** David Bonilla Medina  
**Grau:** Enginyeria Informàtica (Menció Computació) - UAB  
**Curs:** 2025/2026

## 📋 Descripció
Aquest projecte explora l'ús de **Models de Llenguatge Visual (VLMs)** d'última generació (SOTA 2026) i arquitectura *Open Source* per generar descripcions clíniques detallades (*Explainability*) d'imatges de colonoscòpia (pòlips).

S'analitzaran i compararan models com **Qwen3-VL**, **MiniCPM-V 4.5** i **InternVL 3.5**, centrant-se en l'ús de noves tecnologies d'encoder visual i mecanismes de resolució dinàmica.

L'objectiu és demostrar la viabilitat d'executar aquests sistemes en **entorns locals** (Edge AI) utilitzant maquinari de consum, garantint la privacitat de les dades mèdiques mitjançant el backend d'inferència **Ollama**.

## 🛠️ Stack Tecnològic
*   **Llenguatge:** Python 3.12
*   **Gestor de Paquets:** `uv` (Astral)
*   **Backend d'Inferencia:** [Ollama](https://ollama.com/) (Local API)
*   **Llibreries Clau:** `ollama-python`, `pillow`, `requests`, `tqdm`.
*   **Entorn:** Lightweight (sense dependències pesades de PyTorch/Transformers en el llançador).

## 🛠️ Manager Tool (v5.0) - CLI Interactive
El projecte inclou una potent eina de gestió (`setup_env.py`) amb una **Interfície d'Usuari de Text (TUI)** avançada que facilita la configuració, el diagnòstic i el manteniment de l'entorn Ollama.

**Funcionalitats Principals:**
*   **🎮 Interfície Gràfica en Terminal:** Navegació intuïtiva, circular i sense parpadeigs (*Flicker-Free*).
*   **🩺 Diagnòstic Dinàmic:** Analitza l'estat de les llibreries crítiques i la connexió amb el servei Ollama.
*   **🤖 Gestió de Models VLM:** Sistema integrat per descarregar (*pull*) models directament des del registre d'Ollama.
*   **🧪 Smoke Test Interactiu:** Prova d'inferència completa amb selecció de model i validació en temps real.
*   **🔄 Auto-Bootstrapping:** El sistema detecta automàticament si s'està executant fora de l'entorn virtual (`.venv`) i es reinicia dins d'ell per garantir la càrrega de llibreries.

**Controls:**
*   `⬆️` / `⬇️`: Navegar per les opcions.
*   `ESPAI`: Entrar en Submenú / Marcar o desmarcar opcions.
*   `ESC`: Tornar enrere o sortir.
*   `ENTER`: **Confirmar i Executar** la selecció actual.

## 🚀 Instal·lació i Configuració

1.  **Prerequisit: Instal·lar Ollama**
    Descarrega i instal·la Ollama des de [ollama.com](https://ollama.com). Assegura't que el servei estigui actiu (`ollama serve`).

2.  **Clonar el repositori:**
    ```bash
    git clone https://github.com/NIU1635219/TFG_VLM_Medical.git
    cd TFG_VLM_Medical
    ```

3.  **Executar el Manager Tool:**
    ```bash
    # Windows
    .\setup.bat
    ```
    L'script configurarà l'entorn virtual, instal·larà les dependències i obrirà el menú de gestió.

## 📂 Estructura del Projecte
L'arquitectura del projecte està dissenyada per ser modular:
*   `src/inference/`: Controladors d'inferència basats en l'API d'Ollama.
*   `src/scripts/`: Utilitats de terminal (test d'inferència interactiva, etc).
*   `tests/`: Tests unitaris i d'integració (Pytest).
*   `data/`: dataset mèdic segmentat en `raw/` i `processed/`.
*   `setup_env.py`: Script de gestió v5.0 (TUI).

## 🤖 Models VLM Suportats
Actualment, el sistema està optimitzat per als següents models en Ollama:
| Model | Tag en Ollama | Descripció |
| :--- | :--- | :--- |
| **MiniCPM-V 4.5** | `openbmb/minicpm-v4.5:8b` | SOTA OpenBMB (8B) |
| **MiniCPM-V 2.6** | `openbmb/minicpm-v2.6:8b` | Versió Estable (8B) |
| **Qwen3-VL** | `qwen3-vl` | SOTA Razonamiento 2026 (8B) |
| **InternVL 3.5** | `blaifa/InternVL3_5:8b` | InternVL High Performance (8B) |

## 🧪 Testing
Per executar els tests unitaris i verificar la integració amb Ollama:
```bash
uv run python -m pytest tests/
```
