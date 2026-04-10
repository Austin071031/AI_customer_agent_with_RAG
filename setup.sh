#!/bin/bash
# Exit on error
set -e

echo "Setting up AI Customer Agent environment..."

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip to avoid cryptography build errors
echo "Upgrading pip..."
python -m pip install --upgrade pip

# 4. Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! ✅"
echo "To run the application, use:"
echo "  source venv/bin/activate"
echo "  python main.py"
