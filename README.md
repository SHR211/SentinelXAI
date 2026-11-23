# SentinelXAI
### Explainable AI for Intelligent Cyber Risk Assessment

SentinelXAI is a Machine Learning-based cybersecurity application designed to analyze network traffic, identify potential cyberattacks, assess security risks, and provide explainable predictions using SHAP.

The application uses the UNSW-NB15 dataset and provides a web-based Security Operations Center (SOC) dashboard for investigating suspicious network events.

## Key Features

- Upload network traffic CSV datasets.
- Clean, preprocess, and prepare network traffic features.
- Predict potential attacks using three Machine Learning models.
- Calculate weighted ensemble cyber-risk scores.
- Classify events into Low, Medium, High, and Critical severity levels.
- Display security analytics through a SOC dashboard.
- Explain Random Forest predictions using SHAP.
- Investigate individual network events and forensic details.
- Record analyst decisions.
- Generate downloadable PDF security reports.

## Machine Learning Models

The system uses three pretrained classification models:

| Model | Ensemble Weight |
|---|---|
| Random Forest | 40% |
| XGBoost | 40% |
| Logistic Regression | 20% |

**Risk Score = (0.4 × RF Probability + 0.4 × XGBoost Probability + 0.2 × LR Probability) × 100**

The ensemble weights are predefined heuristic values.

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy
- **Explainable AI:** SHAP
- **Model Serialization:** Joblib
- **Report Generation:** ReportLab
- **Dataset:** UNSW-NB15

## Project Workflow

1. Upload the UNSW-NB15 network traffic CSV file.
2. Perform data preprocessing and feature preparation.
3. Generate attack probabilities using pretrained ML models.
4. Combine predictions using a weighted ensemble.
5. Calculate cyber-risk scores and severity levels.
6. Display results in the SOC dashboard.
7. Investigate individual security events.
8. View SHAP explanations and network forensic information.
9. Record analyst decisions and download PDF reports.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/SHR211/SentinelXAI.git
```

2. Open the project directory:

```bash
cd SentinelXAI
```

3. Install required Python packages:

```bash
pip install flask pandas numpy scikit-learn xgboost shap joblib reportlab
```

4. Ensure the required pretrained model files are available in the project directory.

5. Run the Flask application:

```bash
python app.py
```

6. Open your browser:

```text
http://127.0.0.1:5000
```

**Note:** The large pretrained model files `sentinelxai_model.pkl` and `sentinelxai_rf.pkl` are excluded from this repository because of GitHub's standard file-size limit. The application may require these files to run correctly.

## Project Team

**Department of Computer Science and Engineering**

- Karnati Sruthi — U23CS0333
- Karella Vinay — U23CS0329
- Jemallamudi Tejaswi — U23CS0295
- Kandari Mohan — U23CS0320

## Project Scope

SentinelXAI performs offline batch analysis of uploaded network traffic datasets. It is a cybersecurity decision-support prototype, not a real-time network monitoring or intrusion prevention system.

## Conclusion

SentinelXAI combines Machine Learning, explainable AI, and cybersecurity analytics to support transparent network threat assessment and security event investigation.
