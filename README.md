# Loan Approval Prediction – MLOps From Scratch

This project demonstrates an **end‑to‑end MLOps journey**, starting from **data science experimentation** to **production‑ready machine learning**. It is designed as a **beginner‑friendly bootcamp project** with clear separation of responsibilities between a **Data Scientist** and an **MLOps Engineer**.

---

## 🎯 Project Objective

Build a machine learning model that predicts whether a loan application should be **Approved (Y)** or **Rejected (N)** based on applicant information such as income, credit history, education, and property area.

---

## 🧱 Project Structure

```
loan-prediction-project/
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   └── train_simple.py
├── models/
│   └── loan_model.pkl
├── requirements.txt
├── requirements_mlops.txt
├── README.md
└── .gitignore
```

---

## 👩🏽‍🔬 Part 1: Data Scientist Workflow

### 1. Environment Setup

A Python virtual environment is used to isolate dependencies and avoid conflicts.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Generation

Synthetic loan data is generated to simulate a real‑world banking dataset, including:

- Categorical and numerical features
- Missing values
- Realistic approval logic

Script:

```bash
python create_sample_data.py
```

### 3. Exploratory Data Analysis (EDA)

EDA is performed using Jupyter Notebook to:

- Inspect data quality
- Understand feature distributions
- Analyze target imbalance

Notebook:

```
notebooks/01_exploratory_analysis.ipynb
```

### 4. Feature Engineering & Preprocessing

Custom Scikit‑learn transformers are implemented for:

- Mean & mode imputation
- Domain‑based feature engineering
- Encoding categorical variables
- Log transformations

File:

```
src/preprocessing.py
```

### 5. Model Training

A **RandomForestClassifier** is trained using a full preprocessing + modeling pipeline.

```bash
cd src
python train_simple.py
```

Outputs:

- Trained model pipeline
- Validation metrics (Accuracy, F1‑score)
- Serialized model artifact (`.pkl`)

---

## ⚙️ Part 2: MLOps Engineer Workflow

### 1. Separate MLOps Environment

A dedicated environment is created for production and automation tooling.

```bash
python3 -m venv venv-mlops
source venv-mlops/bin/activate
pip install -r requirements_mlops.txt
```

### 2. Production Readiness Goals

This phase focuses on:

- Reproducibility
- Model versioning
- Experiment tracking
- Deployment readiness
- Monitoring & scalability

Tools introduced:

- **MLflow** – experiment tracking
- **DVC** – data & model versioning
- **FastAPI** – model serving
- **Prometheus** – metrics & monitoring
- **AWS S3** – artifact storage

(Implementation continues in later stages of the bootcamp.)

---

## 📦 Artifacts Handed Over

- Training & test datasets
- Feature engineering pipeline
- Training script
- Serialized model
- Dependency files

---

## 🚀 Key Learning Outcomes

- End‑to‑end ML lifecycle understanding
- Clean separation of DS vs MLOps responsibilities
- Production‑grade ML thinking from day one
- Reproducible and scalable ML workflows

---

## 📌 Notes

This project is intentionally simple in modeling but strong in **engineering fundamentals**, making it ideal for:

- Bootcamps
- Junior ML / MLOps roles
- Portfolio demonstrations

---

✅ **Status:** Data Scientist phase completed
➡️ **Next:** Model serving, CI/CD, monitoring, and retraining pipelines
