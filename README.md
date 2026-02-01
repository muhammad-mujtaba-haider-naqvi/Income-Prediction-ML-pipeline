# Income Level Prediction using Machine Learning

## Project Overview
This project implements an end-to-end machine learning pipeline to predict whether an individual’s annual income exceeds $50,000 based on demographic and employment-related attributes. The workflow follows a structured data science approach, including data preprocessing, exploratory data analysis, feature engineering, model training, and comparative evaluation.

The objective of this project is to demonstrate how effective preprocessing and feature engineering techniques can significantly improve model performance and generalization on real-world datasets.

---

## Dataset
- **Name:** Adult Income Dataset (Census Income Dataset)
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/ml/datasets/adult
- **Total Records:** 48,842
- **Target Variable:** Income (`<=50K`, `>50K`)

The dataset consists of both numerical and categorical features such as age, education, occupation, marital status, working hours, and capital gain/loss.


---

## Methodology
The project follows a standard data science pipeline:

1. **Data Loading and Cleaning**
   - Handling missing values
   - Removing inconsistencies

2. **Exploratory Data Analysis (EDA)**
   - Understanding feature distributions
   - Identifying correlations and patterns

3. **Feature Engineering**
   - Domain-based feature creation
   - Correlation-based feature selection
   - Dimensionality reduction using Principal Component Analysis (PCA)

4. **Model Training**
   - Training multiple classification models
   - Applying appropriate preprocessing for each model

5. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix

---

## Machine Learning Models Used
- Logistic Regression
- Naïve Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Artificial Neural Network (ANN)

---

## Results and Key Insights
- Feature engineering significantly improved model stability and generalization.
- PCA reduced multicollinearity and improved performance, especially for linear models.
- Decision Trees showed overfitting before feature selection and dimensionality reduction.
- Ensemble-based and non-linear models performed better on this dataset.
- Random Forest achieved the best overall performance due to its ability to capture complex feature interactions and handle noisy data effectively.

---

## Tools and Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## How to Run the Project
1. Clone the repository:
git clone https://github.com/shaafeadawood/income-prediction-ml.git


2. Navigate to the project directory:
cd income-prediction-ml

3. Install the required dependencies:
pip install -r requirements.txt


4. Open and run the Jupyter Notebook:
notebooks/income_prediction_ml.ipynb

---

## Author
**Muhammad Mujtaba Haider Naqvi**  
Bachelor’s in Computer Science


