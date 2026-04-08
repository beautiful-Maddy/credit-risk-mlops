# Credit Risk Prediction - MLOps Project

## Objective

Predict the probability of a client defaulting on a loan using financial features.

---

## Dataset

* Financial variables: `income`, `fico`, `dti`, etc.
* Target variable: `not.fully.paid`

---

## Models Tested

* Logistic Regression
* Decision Tree
* Random Forest

---

## Experiment Tracking (MLflow)

* **1 model = 1 experiment**
* **Multiple runs = model iterations**

Tracked metrics:

* Precision
* Recall
* F1-score

---

## Selected Model

**Logistic Regression with `class_weight="balanced"`**

### Why this model?

* ✅ Best **recall** → better detection of risky clients
* ✅ Interpretable (important in finance)
* ✅ Good generalization

---

## 📊 Results

| Model               | Recall | Precision |
| ------------------- | ------ | --------- |
| Logistic Regression | 0.67   | 0.26      |
| Decision Tree       | 0.46   | 0.25      |
| Random Forest       | 0.02   | 0.42      |

Focus on **recall** because missing a risky client is more costly than a false positive.

---

## Application (Streamlit)

The app allows:

* Input of client financial data
* Prediction of default risk
* Display of probability score

---

## Tech Stack

* Python (scikit-learn, pandas)
* MLflow (experiment tracking)
* Streamlit (UI)
* Docker (containerization)
* AWS ECS (deployment)
* GitHub Actions (CI/CD)

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

---

## Deployment

The application is deployed on AWS ECS and publicly accessible:

👉 http://13.38.66.245:8501/

---

## CI/CD Pipeline

This project uses **GitHub Actions** to automatically:

* Build Docker image
* Push to Amazon ECR
* Deploy to Amazon ECS

---

## Key Takeaways

* End-to-end MLOps pipeline
* Model tracking and comparison with MLflow
* Automated deployment with CI/CD
* Production-ready ML application

---
