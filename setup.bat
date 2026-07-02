@echo off
REM setup.bat -- one-command setup for Windows
REM Usage: setup.bat

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Running data pipeline...
python src\data\load_data.py
python src\features\feature_engineering.py
python src\models\train_models.py
python -m src.models.error_analysis
python src\explainability\shap_analysis.py
python -m src.models.campaign_manager

echo.
echo Setup complete. Launching dashboard...
streamlit run app.py
