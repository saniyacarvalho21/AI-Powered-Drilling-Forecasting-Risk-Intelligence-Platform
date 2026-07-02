
# 🛢️ AI-Powered Drilling Forecasting & Risk Intelligence Platform

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainable-AI-orange?style=for-the-badge)
![Monte Carlo](https://img.shields.io/badge/Risk-Analysis-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

# Overview

The **AI-Powered Drilling Forecasting & Risk Intelligence Platform** is an industrial-style Decision Support System designed to improve drilling planning through Machine Learning, Explainable AI, and Risk Analytics.

The platform predicts:

- ⏱️ Drilling Duration
- 💰 Total Drilling Cost
- ⚠️ Non-Productive Time (NPT)

Unlike traditional prediction systems, this platform not only generates predictions but also quantifies uncertainty using Monte Carlo Simulation, explains model decisions using SHAP Explainable AI, evaluates model performance through comprehensive error analysis, and provides interactive dashboards for operational decision-making.

The project follows a production-oriented workflow inspired by real-world AI solutions used in drilling analytics and operational intelligence.

---

# Key Features

### Intelligent Prediction Engine

- Predicts Drilling Duration
- Predicts Total Cost
- Predicts NPT (Non-Productive Time)

---

### Multi-Model Machine Learning

The platform automatically trains and compares multiple regression algorithms:

- Linear Regression
- Random Forest
- XGBoost
- CatBoost

The best-performing model is automatically selected based on evaluation metrics.

---

### Risk Intelligence

Monte Carlo Simulation generates:

- P10
- P50
- P90

risk scenarios to quantify uncertainty and support planning decisions.

---

### Explainable AI

The system integrates SHAP to explain:

- Feature importance
- Prediction drivers
- Model reasoning

allowing engineers to understand why predictions were generated.

---

### Error Analysis Dashboard

Performance is evaluated using:

- MAE
- RMSE
- MAPE
- R² Score

Calibration analysis and model comparison dashboards help validate prediction quality.

---

### Interactive Dashboard

Developed using Streamlit and Plotly with a modern control-room interface including:

- Executive KPIs
- Forecast Dashboard
- Monte Carlo Risk Analysis
- Explainability Dashboard
- Error Analysis
- Campaign History
- Recommendation Panel

---

# System Architecture

```
Raw Dataset
      │
      ▼
Data Validation
      │
      ▼
Feature Engineering
      │
      ▼
Multiple ML Models
      │
      ▼
Model Comparison
      │
      ▼
Best Model Selection
      │
      ▼
Prediction Engine
      │
      ▼
Monte Carlo Simulation
      │
      ▼
SHAP Explainability
      │
      ▼
Interactive Dashboard
```

---

# Technology Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- XGBoost
- CatBoost

## Data Processing

- Pandas
- NumPy

## Explainable AI

- SHAP

## Visualization

- Plotly
- Matplotlib

## Scientific Computing

- SciPy

## Web Framework

- Streamlit

---

# Machine Learning Workflow

```
Dataset

↓

Data Validation

↓

Feature Engineering

↓

Train Multiple Models

↓

Model Evaluation

↓

Automatic Best Model Selection

↓

Prediction

↓

Monte Carlo Risk Analysis

↓

SHAP Explainability

↓

Interactive Dashboard
```

---

# Dashboard Modules

## Executive Overview

- Fleet KPIs
- Operational Summary
- Basin Analysis
- Campaign Statistics

---

## Forecasting

Predict:

- Duration
- Cost
- NPT

based on drilling parameters.

---

## Monte Carlo Risk Analysis

Generate uncertainty distributions including:

- P10
- P50
- P90

for operational planning.

---

## Explainable AI

Visualize:

- SHAP Values
- Feature Contributions
- Prediction Drivers

---

## Error Analysis

Model evaluation includes:

- MAE
- RMSE
- MAPE
- R²

with comparison across multiple algorithms.

---

## Campaign History

Track:

- Previous model performance
- Retraining history
- Campaign improvements

---

# Project Structure

```
AI-Powered-Drilling-Forecasting-Risk-Intelligence-Platform/

│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── assets/
│   ├── screenshots/
│   └── architecture/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── src/
│
├── notebooks/
│
└── docs/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI-Powered-Drilling-Forecasting-Risk-Intelligence-Platform.git
```

Move into the project directory

```bash
cd AI-Powered-Drilling-Forecasting-Risk-Intelligence-Platform
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# Screenshots

> Add your dashboard screenshots inside:
<img width="1458" height="796" alt="Screenshot 2026-07-02 at 10 03 41 PM" src="https://github.com/user-attachments/assets/3baded4f-dd25-46e7-acd2-7aff1bfa8605" />
<img width="1057" height="583" alt="Screenshot 2026-07-02 at 10 05 34 PM" src="https://github.com/user-attachments/assets/9422d71f-958a-430f-a1c3-7af201e3ceba" />
<img width="1108" height="770" alt="Screenshot 2026-07-02 at 10 06 21 PM" src="https://github.com/user-attachments/assets/3e8b2faa-7530-4c40-8640-774e537ab31a" />
<img width="1100" height="647" alt="Screenshot 2026-07-02 at 10 06 47 PM" src="https://github.com/user-attachments/assets/b4bcfadf-b0d0-497c-936c-06629a06d546" />
<img width="1086" height="766" alt="Screenshot 2026-07-02 at 10 07 19 PM" src="https://github.com/user-attachments/assets/34f34067-4d78-4015-88aa-ee99630633cb" />
<img width="1080" height="388" alt="Screenshot 2026-07-02 at 10 07 50 PM" src="https://github.com/user-attachments/assets/7bcb6690-8df8-40c1-9bcf-eb70b0096d51" />
<img width="1083" height="511" alt="Screenshot 2026-07-02 at 10 09 04 PM" src="https://github.com/user-attachments/assets/14d4c186-a64b-4462-ae91-536f80ce117f" />




```
assets/screenshots/
```
---

# Future Enhancements

The platform roadmap includes:

- AI Data Quality Engine
- Automated Data Validation
- Drift Detection
- MLOps Pipeline
- Docker Deployment
- Cloud Deployment (AWS / Azure)
- REST API
- Authentication & Role-Based Access
- Batch Prediction
- Real-Time Monitoring
- Digital Twin Integration

---

# Applications

This platform can support:

- Oil & Gas Operators
- Drilling Engineers
- Operations Managers
- Planning Teams
- Data Scientists
- AI Engineers
- Decision Support Teams

---

# Acknowledgements

This project was developed to demonstrate how Machine Learning, Explainable AI, and Risk Intelligence can be integrated into an industrial decision-support platform for drilling operations.

---

# License

This project is licensed under the MIT License.

---

# Contact

**Developer:** Saniya Carvalho

For suggestions, improvements, or collaboration, feel free to connect through GitHub.

---

## ⭐ If you found this project useful, consider giving it a Star.
