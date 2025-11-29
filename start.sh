#!/bin/bash

# Define the name of the virtual environment directory
VENV_DIR="venv"

echo "Stopping any running bot instances..."
pkill -f "python bot.py" || true

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

if [ -f "requirements.txt" ]; then
    echo "Installing/updating requirements..."
    pip install -r requirements.txt --upgrade
fi

echo "Starting bot with fresh instance..."
python bot.py