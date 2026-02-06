import requests

url = "http://localhost:9696/predict"

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
  "ca": 2.0,
}

response = requests.post(url, json = patient)
health = response.json()

print('prob of healthy person =', health)

if health['health_probability'] >= 0.4:
    print('Patient is at heart risk')
else:
    print('Patients heart is healthy')
