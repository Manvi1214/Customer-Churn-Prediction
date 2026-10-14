# Customer Churn Prediction using Machine Learning

A complete, production-ready machine learning system and responsive web application built from scratch to predict customer churn based on subscription, usage, and demographic details. 

This project was built for an internship submission. It features a modern, custom CSS dark-theme user interface (no Bootstrap) and is fully integrated with a Flask backend and a Scikit-learn Random Forest classifier pipeline.

---

## Project Overview

The primary objective is to classify whether a customer will remain active (**Customer Will Stay**) or terminate their subscription (**Customer Will Churn**), alongside the model's prediction confidence probability. 

The model is trained on structured customer records (over 64,000 active test profiles) and evaluates key variables including customer service calls, payment delays, subscription tiers, contract lengths, total spend, age, and tenure.

---

## Key Features

- **End-to-End ML Pipeline:** Data cleaning, missing value imputation, standard scaling, one-hot encoding, and a trained `RandomForestClassifier` ensemble.
- **Explainable Outcomes:** Displays the top 5 global factors contributing to the churn model's decision outcomes.
- **Interactive Predictor Form:** Beautiful responsive form inputs with robust client-side and server-side validation.
- **Color-Coded Alerts:** High-contrast prediction outcome cards (Green for low risk, Red for high risk) featuring animated progress bars.
- **Performance Dashboard:** Visualizes model metrics (Accuracy, Precision, Recall, F1) alongside static evaluation charts (Confusion Matrix, ROC Curve, and Feature Importance).
- **Startup Design System:** Futuristic obsidian-blue glassmorphism theme built completely from scratch using vanilla CSS and custom typography.

---

## Folder Structure

```
Customer-Churn-Prediction/
├── app.py                      # Flask web application entrypoint & routing
├── train_model.py              # ML pipeline: cleaning, training & plotting
├── predict.py                  # Real-time ML inference helper script
├── requirements.txt            # Python dependencies configuration
├── README.md                   # Project documentation and guide
├── dataset/                    # Training and testing raw CSV datasets
│   ├── customer_churn_dataset-testing-master.csv
│   └── customer_churn_dataset-training-master.csv
├── models/                     # Saved ML model artifacts & metrics
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── feature_names.pkl
│   └── metrics.txt
├── static/                     # Static web assets
│   ├── css/
│   │   └── style.css           # Custom CSS styling (no Bootstrap)
│   ├── js/
│   │   └── main.js            # Client-side validation & scroll effects
│   ├── images/                 # Image placeholders
│   └── graphs/                 # Saved model evaluation charts
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       └── roc_curve.png
└── templates/                  # Jinja2 HTML layout pages
    ├── layout.html             # Common HTML layout wrapper
    ├── index.html              # Home/Landing page
    ├── predict.html            # Interactive prediction input form
    ├── result.html             # Color-coded prediction result page
    ├── performance.html        # Model evaluation dashboard page
    └── about.html              # Project documentation page
```

---

## Installation & Setup

1. **Clone or Navigate to the Directory:**
   ```bash
   cd Customer-Churn-Prediction
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

1. **Step 1: Train the Machine Learning Model**
   Run the training script to load the datasets, fit the preprocessors, train the Random Forest Classifier, save models to `models/`, and output evaluation charts to `static/graphs/`.
   ```bash
   python train_model.py
   ```
   *Expected output summary:*
   - Fitting preprocessing pipeline...
   - Training RandomForestClassifier...
   - Accuracy: ~99.8%
   - Saved models and feature names successfully.
   - Saved plots successfully.

2. **Step 2: Start the Web Application**
   Run the Flask server script:
   ```bash
   python app.py
   ```
   *Server details:*
   - Running on local address: `http://localhost:5000` (or `http://127.0.0.1:5000`)
   - Open your web browser and navigate to this URL to view the home page.

---

## Model Evaluation Summary

- **Accuracy Score:** ~99.79%
- **Precision Score:** ~99.87%
- **Recall Score:** ~99.69%
- **F1-Score:** ~99.78%
- **Key Predictors:** Payment Delay and Support Calls account for over 60% of the model split importances.

---

## Future Improvements

- **Interactive Explanations:** Integrate SHAP to display customer-specific feature contributions dynamically on the result page.
- **Relational Databases:** Upgrade from CSV-only files to PostgreSQL for secure transaction storage.
- **AutoML Retraining:** Deploy Apache Airflow schedules to retrain the model automatically as new data is uploaded.
