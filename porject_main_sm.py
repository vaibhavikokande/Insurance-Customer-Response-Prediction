# Import necessary libraries for data manipulation (pandas, numpy), preprocessing (LabelEncoder, MinMaxScaler, SimpleImputer), handling class imbalance (SMOTE),
# machine learning (XGBClassifier), and saving the model (joblib).

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier
from joblib import dump
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.impute import SimpleImputer

# Load data
insurance_cust = pd.read_csv(r"data.csv")

# Drop unnecessary column
insurance_cust.drop(columns=["id"], axis=1, inplace=True)

#Converts Vehicle_Damage into numerical values: Yes ? 1, No ? 0
insurance_cust["Vehicle_Damage"].replace(to_replace="Yes",value=1,inplace=True)
insurance_cust["Vehicle_Damage"].replace(to_replace="No",value=0,inplace=True)

#Encodes Gender and Vehicle_Age using LabelEncoder to turn categorical values into numbers (e.g., Male/Female ? 1/0)
label_encoder = LabelEncoder()
insurance_cust["Gender"] = label_encoder.fit_transform(insurance_cust["Gender"])# Male=1, Female=0
insurance_cust["Vehicle_Age"] = label_encoder.fit_transform(insurance_cust["Vehicle_Age"])


insurance_cust_dummies=pd.get_dummies(insurance_cust)
# Creates dummy variables for categorical features, turning them into a one-hot encoded format.

x = insurance_cust.drop(['Response'], axis = 1)
y = insurance_cust.loc[:,'Response'].values
# Separates the dataset into x (features) and y (target variable Response)

x.dropna(inplace=True)
# Drops rows in x with missing values.

imputer = SimpleImputer(strategy='mean') # Choose a strategy: 'mean', 'median', 'most_frequent'


# Impute missing values in 'x' before dropping rows in 'y'
x = imputer.fit_transform(x)

# Now, align 'y' with 'x' after imputation:
x = pd.DataFrame(x)  # Convert x back to DataFrame for indexing
x.reset_index(drop=True, inplace=True) # Reset index of x
# Converts x back to a DataFrame and resets its index
y = y[x.index]  # Ensures y aligns with x after imputation.

# Now apply SMOTE
sm = SMOTE(random_state=42)
# Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset by oversampling the minority class.

# Use 'x' instead of 'X' as input to fit_resample
x_res, y_res = sm.fit_resample(x, y)

# Train the Model
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_res, y_res)
# Trains the model using the oversampled dataset (x_res, y_res).


# Save the trained model
dump(model, 'insurance_model_sm.joblib')  # Saves the trained model to a file (insurance_model_sm.joblib) for later use.
