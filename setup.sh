#!/bin/bash
# setup.sh -- one-command setup for Mac/Linux
# Usage: bash setup.sh

set -e

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "Running data pipeline..."
python src/data/load_data.py
python src/features/feature_engineering.py
python src/models/train_models.py
python -m src.models.error_analysis
python src/explainability/shap_analysis.py
python -m src.models.campaign_manager  # seeds campaign history baseline

echo ""
echo "Setup complete. Launching dashboard..."
streamlit run app.py
