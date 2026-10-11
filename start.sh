#!/usr/bin/env bash
# Create a virtualenv if needed, install dependencies, then print how to start.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  VENV=".venv"
elif [ -d "venv" ]; then
  VENV="venv"
else
  echo "Creating .venv..."
  if command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    python3 -m venv .venv
  fi
  VENV=".venv"
fi

if [ -f "$VENV/Scripts/activate" ]; then
  # Windows Git Bash / MSYS
  # shellcheck disable=SC1091
  source "$VENV/Scripts/activate"
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  cp config/.env.example .env
  echo "Created .env from config/.env.example"
fi

echo
echo "Setup complete."
echo "Start the server with: python app.py"
echo "Then open: http://localhost:3000"
