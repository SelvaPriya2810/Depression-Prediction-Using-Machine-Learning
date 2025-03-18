# Depression Prediction Using Machine Learning

## 📌 Project Overview
This project aims to predict **depression risk** based on various factors like **age, profession, financial stress, and academic pressure** using **Machine Learning models**. The dataset is sourced from **Kaggle** and preprocessed, encoded, and scaled to ensure optimal model performance.

---
## 🚀 Features
- **Data Preprocessing**: Encoding categorical variables using **One-Hot Encoding, Frequency Encoding, and Target Encoding**.
- **Feature Scaling**: Standardization (Z-score) applied to numerical features.
- **Model Training**: Various **ML models** implemented for classification.
- **Model Evaluation**: Accuracy, Precision, Recall, and F1-score compared.

---
## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy** (Data Manipulation)
- **Scikit-learn** (Machine Learning Models)
- **XGBoost** (Boosting Model)
- **Matplotlib, Seaborn** (Visualization)

---
## 📂 Project Structure
```
├── data/               # Dataset (sourced from Kaggle)
├── models/             # Saved trained models
├── scripts/            # Python scripts for preprocessing & training
├── notebooks/          # Jupyter/Colab notebooks
├── README.md           # Project Documentation (This File)
├── requirements.txt    # Dependencies
```

---
## 📊 Machine Learning Models Applied
| Model | Description |
|--------|-------------|
| **Logistic Regression** | Baseline model for binary classification |
| **Random Forest** | Handles feature interactions & works well with mixed data |
| **SVM** | Best for non-linear decision boundaries |
| **XGBoost** | High-performance boosting model |
| **KNN** | Simple but effective for well-separated data |

---
## 🔧 How to Run the Project
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo/depression-prediction.git
   cd depression-prediction
   ```
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Download the dataset from Kaggle**:
   - Visit the Kaggle dataset page: [Kaggle Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset) (replace with actual link)
   - Download the dataset and place it inside the `data/` folder
   
4. **Run the model training script**:
   ```sh
   python scripts/train_model.py
   ```
5. **Evaluate the model**:
   ```sh
   python scripts/evaluate.py
   ```

---
## 📈 Results & Insights
- **Feature Importance Analysis** shows that **financial stress** and **academic pressure** are key predictors.
- **XGBoost performed best** with the highest accuracy and F1-score.
- **Further improvements** can be made by **hyperparameter tuning** and using more advanced deep learning techniques.



