Metabolic Syndrome Prediction Using Machine Learning
This repository contains a Python script and a Jupyter Notebook for predicting the presence of Metabolic Syndrome using clinical and laboratory data. Metabolic Syndrome is a complex medical condition associated with an increased risk of cardiovascular diseases and type 2 diabetes. The project involves data preprocessing, feature engineering, and the implementation of multiple machine learning models to classify individuals as having or not having the syndrome.

Key Features
Data Preprocessing: Handles missing values, encodes categorical variables, and scales numerical features.
Model Implementation: Includes K-Nearest Neighbors (KNN), Decision Tree, Logistic Regression, Random Forest, and Support Vector Classifier (SVC).
Hyperparameter Tuning: Uses GridSearchCV and RandomizedSearchCV to optimize model performance.
Evaluation Metrics: Reports accuracy, precision, recall, and F1-score for each model.
Tools and Technologies
Programming Language: Python
Libraries: Pandas, NumPy, Scikit-learn
Data Handling: CSV file input/output
Model Evaluation: Classification metrics (accuracy, precision, recall, F1-score)
Notebook Support: Jupyter Notebook (.ipynb) for interactive analysis and visualization.
Dataset
The dataset used in this project contains clinical and laboratory measurements such as:

Age, Sex, Marital Status, Income, Race
Waist Circumference, BMI, Albuminuria, Blood Glucose, HDL, Triglycerides
Target Variable: Metabolic Syndrome (1 = Present, 0 = Absent)
How to Use
Clone this repository: git clone https://github.com/SoulNaD0/Metabolic-Syndrome-Prediction.git

Run the Python script or explore the project interactively with the Jupyter Notebook:

To open the notebook, use: jupyter notebook metabolic_syndrome_prediction.ipynb
Requirements
Python 3.x
Jupyter Notebook (for .ipynb version)
Required libraries are listed in requirements.txt
