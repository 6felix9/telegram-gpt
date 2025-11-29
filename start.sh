#!/bin/bash

# Define the name of the virtual environment directory
VENV_DIR="venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    
    echo "Activating virtual environment..."
    source $VENV_DIR/bin/activate
    
    if [ -f "requirements.txt" ]; then
        echo "Installing requirements..."
        pip install -r requirements.txt
    fi
else
    echo "Virtual environment found. Activating..."
    source $VENV_DIR/bin/activate
fi

echo "Starting bot..."
python bot.py
