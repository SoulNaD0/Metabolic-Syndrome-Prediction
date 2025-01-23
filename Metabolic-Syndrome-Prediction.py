# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Load the dataset
raw_df = pd.read_csv("Metabolic Syndrome.csv")

# Display dataset information and check for duplicates
print(raw_df.info())
print(f"\nThe number of duplicated rows: {raw_df.duplicated().sum()}")

# Separate categorical and numerical data
cat_data = raw_df.select_dtypes('object')
num_data = raw_df.select_dtypes(['float64', 'int64']).iloc[:, 1:-1]
y = raw_df['MetabolicSyndrome']

# Handle missing values in categorical data using SimpleImputer
si = SimpleImputer(strategy='most_frequent')
cat_imp = si.fit_transform(cat_data)
catimp = pd.DataFrame(cat_imp, columns=cat_data.columns)

# Encode categorical data using LabelEncoder and OneHotEncoder
le = LabelEncoder()
cat_data_bin = le.fit_transform(cat_data['Sex'])
cat_data_bin = pd.DataFrame(cat_data_bin, columns=['Sex'])

ohe = OneHotEncoder()
ohe_data = ohe.fit_transform(catimp[['Marital', 'Race']]).toarray()

# Combine encoded categorical data, numerical data, and target variable
final = np.concatenate([cat_data_bin.values, num_data.values, ohe_data], axis=1)
final = pd.DataFrame(final)

# Split the data into training and testing sets
x = final.values
y = y.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=444)

# Scale the data using RobustScaler
scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Handle missing values in scaled data using KNNImputer
kni = KNNImputer(n_neighbors=5)
x_train_imp = kni.fit_transform(x_train_scaled)
x_test_imp = kni.transform(x_test_scaled)

# K-Nearest Neighbors (KNN) Model
knc = KNeighborsClassifier()
params = {
    'n_neighbors': range(1, 30, 2),
    'weights': ['uniform', 'distance']
}
gclf = GridSearchCV(knc, param_grid=params, cv=5, scoring='f1')
gclf.fit(x_train_imp, y_train)
print(f"Best params for KNN: {gclf.best_params_}")

best_model = gclf.best_estimator_
y_pred = best_model.predict(x_test_imp)
print(f"KNN Test Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Support Vector Classifier (SVC) Model
svclass = SVC(class_weight='balanced')
params_svc = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['poly', 'rbf'],
    'gamma': [0.001, 0.01, 0.1, 1]
}
svgrid = RandomizedSearchCV(svclass, param_distributions=params_svc, cv=5, scoring='f1', random_state=444, n_iter=5)
svgrid.fit(x_train_imp, y_train)
print(f"Best params for SVC: {svgrid.best_params_}")

best_svc_model = svgrid.best_estimator_
y_svc_pred = best_svc_model.predict(x_test_imp)
print(f"SVC Test Accuracy: {accuracy_score(y_test, y_svc_pred)}")
print(classification_report(y_test, y_svc_pred))

# Decision Tree Model
dte = DecisionTreeClassifier(class_weight='balanced')
params_dte = {
    'max_depth': range(3, 10),
    'max_leaf_nodes': range(3, 10),
    'criterion': ['gini', 'entropy']
}
tgs = GridSearchCV(dte, param_grid=params_dte, cv=5, scoring='f1')
tgs.fit(x_train_imp, y_train)
print(f"Best params for Decision Tree: {tgs.best_params_}")

best_dt_model = tgs.best_estimator_
y_pred_dt = best_dt_model.predict(x_test_imp)
print(f"Decision Tree Test Accuracy: {accuracy_score(y_test, y_pred_dt)}")
print(classification_report(y_test, y_pred_dt))

# Logistic Regression Model
log_reg = LogisticRegression(class_weight='balanced')
params_log_reg = {
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
    'max_iter': [200]
}
tgs_log_reg = GridSearchCV(log_reg, param_grid=params_log_reg, cv=5, scoring='f1')
tgs_log_reg.fit(x_train_imp, y_train)
print(f"Best params for Logistic Regression: {tgs_log_reg.best_params_}")

best_lr_model = tgs_log_reg.best_estimator_
y_pred_lr = best_lr_model.predict(x_test_imp)
print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print(classification_report(y_test, y_pred_lr))

# Random Forest Model
rf = RandomForestClassifier(class_weight='balanced')
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [2, 3],
    'max_features': ['log2', 'sqrt'],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [4, 6],
    'min_samples_leaf': [15, 16],
    'min_samples_split': [15, 16]
}
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=444)
rfclf = GridSearchCV(rf, param_grid=param_grid_rf, cv=stratified_kfold, scoring='f1')
rfclf.fit(x_train_imp, y_train)
print(f"Best params for Random Forest: {rfclf.best_params_}")

best_rf_model = rfclf.best_estimator_
y_pred_rf = best_rf_model.predict(x_test_imp)
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(classification_report(y_test, y_pred_rf))