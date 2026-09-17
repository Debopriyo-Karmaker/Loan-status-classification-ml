# Credit Risk Assessment & Loan Status Prediction

This repository contains a comprehensive machine learning pipeline designed to predict loan approval outcomes using a financial dataset. The project automates credit risk assessment by evaluating an applicant's personal and financial background to determine their likelihood of default, minimizing institutional financial loss.

This project was developed for **CSE 422: Artificial Intelligence** (Section 14) at BRAC University.

## Authors
* **Debopriyo Karmaker**
* **Talha Bin Zakir** 

## Dataset Details
The predictive models were trained and evaluated using a financial dataset (`14.csv`) consisting of **45,000 data points and 14 distinct features**[cite: 2]. 
* **Target Variable:** `loan_status` (Discrete, Binary: 0 = Denied/Default, 1 = Approved).
* **Feature Types:** A heterogeneous mix of continuous numerical data (e.g., `person_income`, `loan_amnt`), discrete numerical data (e.g., `person_age`), and categorical text data (e.g., `person_home_ownership`).

## Machine Learning Pipeline
The project workflow is structured into four primary phases:

1. **Exploratory Data Analysis (EDA):** Generated correlation matrices and heatmaps to identify multicollinearity and measure linear dependencies between financial attributes[cite: 2].
2. **Data Preprocessing:**
   * **Imputation:** Handled missing dataset values by applying mean imputation for numerical columns and mode imputation for categorical columns.
   * **Encoding:** Applied Scikit-Learn's `LabelEncoder` to convert categorical text features into machine-readable integers[cite: 2].
   * **Scaling:** Applied `StandardScaler` to remove distance bias and standardize continuous features to unit variance[cite: 2].
3. **Model Training:** Split the data using an 80/20 train-test ratio. Deployed three supervised models and one unsupervised model:
   * Logistic Regression
   * Decision Tree Classifier
   * Neural Network (`MLPClassifier` with hidden layers of 32 and 16 neurons)[cite: 2]
   * K-Means Clustering (`k=2` for unsupervised grouping)[cite: 2]
4. **Evaluation:** Evaluated supervised models against the holdout test set using Accuracy, Precision, Recall, F1 Score, Confusion Matrices, and ROC/AUC curves.

## Performance Results

The Neural Network (MLP) significantly outperformed the baseline algorithms, demonstrating a superior ability to map the complex, multi-dimensional, and non-linear interactions inherent in human financial data[cite: 2].

| Model | Accuracy | Precision | F1 Score | AUC Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8901 | 0.7663 | 0.7482 | 0.9483 |
| **Decision Tree** | 0.8984 | 0.7740 | 0.7721 | 0.8527 |
| **Neural Network (MLP)** | **0.9153** | **0.8728** | **0.7932** | **0.9660** |

*Metrics based on a 20% holdout test set (random_state=42).*

## Technologies Used
* **Python 3**
* **Pandas** (Data manipulation and imputation)
* **Scikit-Learn** (Preprocessing, Model implementation, Evaluation metrics)
* **Matplotlib** (Data visualization, Heatmaps, ROC plotting)
* **LaTeX** (IEEE-formatted academic reporting)
