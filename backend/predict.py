import joblib, sys, json
import numpy as np

model = joblib.load('loan_model.pkl')
encoders = joblib.load('label_encoders.pkl')
features = joblib.load('features_list.pkl')

data = json.loads(sys.stdin.read())

data['ApplicantIncome'] = int(data['ApplicantIncome'])
data['CoapplicantIncome'] = float(data['CoapplicantIncome'])
data['LoanAmount'] = float(data['LoanAmount'])
data['Loan_Amount_Term'] = float(data['Loan_Amount_Term'])
data['Credit_History'] = int(data['Credit_History'])


for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
    data[col] = encoders[col].transform([data[col]])[0]

data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
data['LoanAmount_log'] = np.log(data['LoanAmount'] + 1)
data['EMI'] = data['LoanAmount'] / data['Loan_Amount_Term']
data['Balance_Income'] = data['TotalIncome'] - (data['EMI'] * 1000)

X = [[data[col] for col in features]]

prediction = model.predict(X)[0]
proba = model.predict_proba(X)[0][1] * 100

print(json.dumps({
    'prediction': str(prediction),
    'probability': float(proba)})
)

#logging
# print("Data before prediction:", data)
# print("Encoded input:", X)

# data = json.loads(sys.stdin.read())
# print("Incoming data from frontend:", data, file=sys.stderr)
