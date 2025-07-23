from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

df = pd.read_csv('/path/loan_data.csv')
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
  
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['Balance_Income'] = df['TotalIncome'] - ((df['LoanAmount'] * 100000) / df['Loan_Amount_Term'])

X = df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'TotalIncome',
        'LoanAmount_log', 'Balance_Income']]
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(classification_report(y_test, y_pred))

import joblib

# Save model
joblib.dump(model, 'loan_model.pkl')

# Save encoders dictionary
joblib.dump(label_encoders, 'label_encoders.pkl')

# Optional: Save the list of features used during training
features_used = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
                 'TotalIncome', 'LoanAmount_log', 'Balance_Income']
joblib.dump(features_used, 'features_list.pkl')
