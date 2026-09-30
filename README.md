# Mango FPO AI Decision Support System

AI-Based Decision Support System for Mango Farmer Producer Organizations (FPOs).

This is a runnable MVP/prototype. It includes a Streamlit dashboard, a demo dataset, and a Random Forest price-prediction model. The demo dataset is SYNTHETIC and must be replaced with verified real mango data before academic results are reported.

## Run on Mac
```bash
cd mango_fpo_ai_dss
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_demo_data.py
python src/train_model.py
streamlit run app.py
```

The dashboard includes price prediction, market comparison, revenue estimation, and a transparent decision-support message.

## Final-project direction
Replace demo data with verified official data; add time-aware validation, model comparison, weather/production integration, transport/storage costs, explainability, testing, deployment and research documentation.
