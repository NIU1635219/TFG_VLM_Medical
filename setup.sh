#!/bin/bash
set -euo pipefail

if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 could not be found. Please install Python 3.12."
    exit 1
fi

echo "[SETUP] Launching automation script..."
if ! python3 setup_env.py; then
    echo "[ERROR] Setup failed. Review errors above."
    exit 1
fi

echo "[SUCCESS] Environment is ready."
