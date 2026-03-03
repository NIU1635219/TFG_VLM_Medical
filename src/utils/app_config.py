"""Constantes de configuración estáticas de la aplicación.

Este módulo centraliza los registros y mapas de dependencias para que no
residan en el punto de entrada ``setup_env.py``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Registro de modelos VLM para descarga / pull
# ---------------------------------------------------------------------------

MODELS_REGISTRY: dict[str, dict] = {
    "minicpm_v_2_6_8b": {
        "name": "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf",
        "description": "MiniCPM-V 2.6 (8B) - Versión Estable Compatible",
    },
    "minicpm_v_4_5_8b": {
        "name": "https://huggingface.co/openbmb/MiniCPM-V-4_5-gguf",
        "description": "MiniCPM-V 4.5 (8B) - SOTA OpenBMB",
    },
    "qwen3_vl_8b": {
        "name": "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF",
        "description": "Qwen3-VL 8B (SOTA Razonamiento 2026)",
    },
    "internvl3_5_8b": {
        "name": "https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-8B-GGUF",
        "description": "InternVL 3.5 (8B) - Blaifa",
    },
    "internvl3_5_14b": {
        "name": "https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-14B-GGUF",
        "description": "InternVL 3.5 (14B) - OpenGVLab (GGUF)",
    },
    "qwen3_vl_8b_thinking": {
        "name": "https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking-GGUF",
        "description": "Qwen3-VL 8B Thinking (GGUF)",
    },
}

# ---------------------------------------------------------------------------
# Dependencias Python requeridas
# ---------------------------------------------------------------------------

REQUIRED_LIBS: list[str] = [
    "lmstudio",
    "pydantic",
    "pillow",           # Import name: PIL
    "opencv-python",
    "pandas",
    "matplotlib",
    "numpy",
    "rarfile",
    "jupyter",
    "pytest",
    "tqdm",
]

# Mapeo paquete→módulo de importación (solo cuando difieren)
LIB_IMPORT_MAP: dict[str, str] = {
    "lmstudio": "lmstudio",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "pyyaml": "yaml",
    "pytest-mock": "pytest_mock",
}
