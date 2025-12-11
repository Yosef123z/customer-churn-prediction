# 📉 Customer Churn Prediction

[![Python 3.x](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Machine Learning solution to predict customer churn using demographic and service usage data. This project identifies customers likely to stop using a service, enabling businesses to take proactive retention measures.

---

## 📊 Project Overview

Customer churn is a critical metric for any subscription-based business. This project builds a predictive model to classify customers as "Churn" (likely to leave) or "No Churn" (likely to stay).

### **The Challenge: "Inverted" Class Imbalance**

The provided dataset presented a unique statistical anomaly:

- **Churn (Yes):** ~88%
- **No Churn (No):** ~12%

In real-world scenarios, churners are usually the minority. This extreme "inverted" imbalance meant a standard model would simply predict "Churn" for everyone and achieve 88% accuracy without learning anything.

### **The Solution: Strategic Undersampling**

To fix this, I implemented a **2:1 Undersampling Strategy**:

1. **Isolated the Minority Class:** Preserved all "No Churn" records.
2. **Sampled the Majority Class:** Randomly selected "Yes" records to create a roughly 2:1 ratio (67% Yes / 33% No).
3. **Result:** This balanced the training data enough for the model to learn the distinct patterns of loyal customers, preventing it from blindly guessing "Churn."

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Random Forest, Logistic Regression, KNN, SVM) |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit |
| **Model Persistence** | Joblib |

---

## 📂 Project Structure

```text
├── app.py                   # Streamlit web application
├── customer_churn.ipynb     # Jupyter Notebook for EDA and Model Training
├── customer_churn_data.csv  # Dataset
├── model.pkl                # Trained Random Forest model
├── scaler.pkl               # Fitted Standard Scaler
├── requirements.txt         # List of dependencies
├── Proposal.md              # Project proposal document
├── README.md                # Project documentation
├── LICENSE                  # MIT License
└── .gitignore               # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

This will launch the web application in your default browser. Enter customer details to get a churn prediction.

### Running the Jupyter Notebook

```bash
jupyter notebook customer_churn.ipynb
```

The notebook contains the complete ML pipeline including:
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Model Training & Evaluation
- Model Export

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 84.5% | Baseline model |
| K-Nearest Neighbors | 88.5% | n_neighbors=3, weights=distance |
| Support Vector Machine | 89.5% | C=10, kernel=rbf |
| **Random Forest** | **89.5%** | **Selected model** (n_estimators=200) |

### Best Model: Random Forest Classifier

- **Accuracy:** 89.5%
- **Cross-Validation Mean Accuracy:** 95.04%
- **Minority Class (No Churn) Recall:** 78%

The Random Forest model was selected for deployment due to its consistent performance and robustness against overfitting.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Author
- **Youssef Zannon**

