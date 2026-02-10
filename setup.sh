#!/bin/bash
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 could not be found. Please install Python 3.12."
    exit 1
fi

echo "[SETUP] Launching automation script..."
python3 setup_env.py
