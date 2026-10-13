#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    echo 'Primero creá el entorno: python3 -m venv .venv' >&2
    exit 1
fi
"$project_root/.venv/bin/python" -m pip install -e "$project_root" --no-deps
echo 'Comando instalado. Con .venv activado, usá: client --help'
