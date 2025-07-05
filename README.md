# 📉 Customer Churn Prediction using PySpark & Machine Learning

This project aims to predict **customer churn** — identifying customers likely to discontinue a service — using big data processing with **Apache Spark** and machine learning models built on **PySpark MLlib**.  
Data analysis and visualization are further enhanced using **Pandas, NumPy, Matplotlib, and Seaborn**.

---

##  Tech Stack & Tools Used
| Tool / Library                  | Purpose                                  |
|----------------------------------|------------------------------------------|
| **PySpark (Spark SQL + MLlib)**  | Distributed data processing & ML pipeline |
| **Pandas, NumPy**                | Data manipulation, local exploration      |
| **Matplotlib, Seaborn**          | Data visualization & plots                |
| **Google Colab / Jupyter**       | Interactive Python environment            |
| **Pipeline, CrossValidator**     | For building and tuning ML workflows      |

---

##  Project Highlights
- **Exploratory Data Analysis (EDA)** using Spark DataFrames, visualized with Matplotlib & Seaborn.
- Applied feature engineering including:
  - Null & anomaly handling
  - Vector Assembler for combining features
  - Standard & MinMax scaling
- Built multiple ML models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient-Boosted Trees (GBT)
- Used `CrossValidator` with `ParamGridBuilder` for hyperparameter tuning.
- Evaluated with metrics like Accuracy, F1 Score, and Confusion Matrix.

---

## 🔗 Full Dataset
Because of GitHub file size limits, the complete dataset is not uploaded here.

- 📂 Download the full dataset from: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
- Place it in the `data/` directory as `churn_full.csv` before running the notebook.



##  Results
- Identified top features influencing churn, such as contract type and monthly charges.
- Achieved up to:
  - **Accuracy:** ~87%
  - **ROC AUC:** ~0.91
- Visualization of churn distribution and feature correlations provided actionable insights.

  ![image](https://github.com/user-attachments/assets/f95d901c-15e6-45c4-8057-733e744ece07)
![image](https://github.com/user-attachments/assets/d6f49fa9-2474-45c7-8657-c700c1f330f6)
![image](https://github.com/user-attachments/assets/5aae5197-16ba-4f6b-844c-6f2555f541ad)
![image](https://github.com/user-attachments/assets/5e8c1d4d-9f97-4fb2-a669-691d705d4204)


---

##  How to Run
1. Clone this repo.
2. Open the notebook in **Google Colab** (or your local Jupyter with PySpark configured).
3. Upload your dataset (CSV or Parquet).
4. Execute cells to process data and run models.

---

## Future Improvements
- Integrate Spark Streaming for real-time churn scoring.
- Store trained models using `MLlib`’s model save/load APIs for deployment.

---

##  Author
**Muskan Meena**  
[LinkedIn(MUSKANMEENA)] | [GitHub](Muskan9114
)

---

##  License
Distributed under the MIT License. See [`LICENSE`](./LICENSE) for more information.
