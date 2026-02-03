import pickle





model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

patient = {
  "sex": "female",
  "cp": "atypical_angina",
  "fbs": "false",
  "restecg": "normal",
  "exang": "false",
  "slope": "upsloping",
  "thal": "normal",
  "age": 71,
  "trestbps": 160.0,
  "chol": 302.0,
  "thalch": 162.0,
  "oldpeak": 0.4,
  "ca": 2.0
}

prob = pipeline.predict_proba(patient)[0, 1]
print('probability of health patient = ', prob)

if prob >= 0.4:
    print('Patient is at heart risk')
else:
    print('Patients heart is healthy')
