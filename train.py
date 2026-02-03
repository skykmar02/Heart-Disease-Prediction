import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

import pickle

# --------------------
# Parameters
# --------------------
C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

numerical = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

categorical = [
    'sex', 'cp', 'fbs', 'restecg',
    'exang', 'slope', 'thal'
]

# --------------------
# Data loading & prep
# --------------------
def load_data(path="data.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    df["fbs"] = df["fbs"].map({True: 'true', False: 'false'})
    df["exang"] = df["exang"].map({True: 'true', False: 'false'})

    for c in df.dtypes[df.dtypes == 'object'].index:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df["target"] = (df['num'] > 0).astype(int)

    df = df.drop(columns=['num', 'id', 'dataset'])

    for c in numerical:
        df[c] = df[c].fillna(df[c].median())

    for c in categorical:
        df[c] = df[c].fillna(df[c].mode()[0])

    return df

# --------------------
# Training
# --------------------
def train(df_train, y_train, C):
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(C=C, solver='liblinear', max_iter=10000)
    )

    pipeline.fit(train_dicts, y_train)
    return pipeline

def predict(df, pipeline):
    dicts = df[categorical + numerical].to_dict(orient='records')
    return pipeline.predict_proba(dicts)[:, 1]

# --------------------
# Cross-validation
# --------------------
def cross_validate(df, C, n_splits):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        y_train = df_train.target.values
        y_val = df_val.target.values

        pipeline = train(df_train, y_train, C)
        y_pred = predict(df_val, pipeline)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

        print(f'fold {fold}: auc={auc:.3f}')

    print(f'CV result: C={C} {np.mean(scores):.3f} ± {np.std(scores):.3f}')
    return scores

# --------------------
# Save model
# --------------------
def save_model(pipeline, output_file):
    with open(output_file, 'wb') as f_out:
        pickle.dump((pipeline), f_out)

    print(f'the model is saved to {output_file}')


df = load_data()

df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1
)

print("Running cross-validation")
cross_validate(df_full_train, C, n_splits)

print("Training final model")
pipeline = train(
    df_full_train,
    df_full_train.target.values,
    C
)

y_pred = predict(df_test, pipeline)
auc = roc_auc_score(df_test.target.values, y_pred)
print(f'Test AUC = {auc:.3f}')

save_model(pipeline, output_file)  
