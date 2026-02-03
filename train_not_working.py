

'''import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from sklearn.pipeline import make_pipeline

import pickle


#Parameters
C = 1.0
n_splits = 5
output_file = f'model_C = {C}.bin'


numerical = ['age',
    'trestbps',
    'chol',
    'thalch',
    'oldpeak',
    'ca'
    ]

categorical = [
    'sex',
    'cp',
    'fbs',
    'restecg',
    'exang',
    'slope',
    'thal',
    ] 


# Data Prep

def load_data():
    df = pd.read_csv("data.csv")
    df.columns = df.columns.str.lower().str.replace(" ","_")

    df["fbs"] = df["fbs"].map({True: 'true', False: 'false'})
    df["exang"] = df["exang"].map({True: 'true', False: 'false'})

    string = list(df.dtypes[df.dtypes == 'object'].index)

    for c in string:
        df[c] = df[c].str.lower().str.replace(' ','_')

    df["target"] = (df['num']>0).astype(int)

    del df['num']
    del df['id']
    del df['dataset']

    for n in numerical:
        df[n] = df[n].fillna(df[n].median())

    for n in categorical:
        df[n] = df[n].fillna(df[n].mode()[0])

    return df



df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)

df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)


#Training

def train(df_train, y_train, C= 0.1):
    #dv = DictVectorizer(sparse = False)
    train_dicts= df_train[categorical + numerical].to_dict(orient = 'records')

    pipeline = make_pipeline(
    DictVectorizer(sparse = False),
    LogisticRegression(C=C, solver = 'liblinear', max_iter=10000))
    

    
    #X_train = dv.fit_transform(train_dicts)

    #model = LogisticRegression(C=C, max_iter=10000)
    pipeline.fit(train_dicts, y_train)

    return pipeline

def predict(df, pipeline):
    dicts= df[categorical + numerical].to_dict(orient = 'records')
    #X = dv.transform(dicts)

    y_pred = pipeline.predict_proba(dicts)[:, 1]

    return y_pred


#Validation

print(f'doing validation with C = {C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold =0
for train_idx, val_idx  in  kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.target.values
    y_val = df_val.target.values

    pipeline  = train(df_train,  y_train, C=C)
    y_pred = predict(df_val, pipeline )

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1
    
print('validation result:')
print(f'C={C} {np.mean(scores).round(3)}, {np.std(scores).round(3)}')


#Training the final model

pipeline  = train(df_full_train, df_full_train.target.values, C=0.1)
y_pred = predict(df_test, pipeline )

y_test = df_test.target.values
auc = roc_auc_score(y_test, y_pred)
print(f'auc = {auc}')


#Save the model using Pickel
def save_model(
        
)
with open(output_file, 'wb') as f_out:
    pickle.dump((pipeline), f_out)

print(f'the model is saved to {output_file}')

df = load_data()
pipeline = train(df)
'''



