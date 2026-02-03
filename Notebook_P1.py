#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


# In[2]:


df = pd.read_csv("data.csv")
df.columns = df.columns.str.lower().str.replace(" ","_")

df["fbs"] = df["fbs"].map({True: 'true', False: 'false'})
df["exang"] = df["exang"].map({True: 'true', False: 'false'})

string = list(df.dtypes[df.dtypes == 'object'].index)

for c in string:
    df[c] = df[c].str.lower().str.replace(' ','_')

df["target"] = (df['num']>0).astype(int)


# In[3]:


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


# In[4]:


del df['num']
del df['id']
del df['dataset']


# In[5]:


for n in numerical:
    df[n] = df[n].fillna(df[n].median())

for n in categorical:
    df[n] = df[n].fillna(df[n].mode()[0])


# In[6]:


df.isnull().sum()


# In[7]:


df


# In[8]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)


# In[9]:


df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)


# In[10]:


len(df_train), len(df_val), len(df_test)


# In[11]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_full_train = df_full_train.reset_index(drop=True)


# In[12]:


y_train = df_train['target']
y_val = df_val['target']
y_test = df_test['target']

del df_train['target']
del df_val['target']
del df_test['target']


# In[13]:


y_val


# In[14]:


df_train


# In[15]:


dv = DictVectorizer(sparse = False)


# In[16]:


train_dicts= df_train[categorical + numerical].to_dict(orient = 'records')
val_dicts= df_val[categorical + numerical].to_dict(orient = 'records')


# In[17]:


dv.fit(train_dicts)


# In[18]:


dv.get_feature_names_out()


# In[19]:


dv.transform(train_dicts[:5])


# In[20]:


dv.transform(val_dicts[:5])


# In[21]:


X_train = dv.fit_transform(train_dicts)


# In[22]:


X_val = dv.transform(val_dicts)


# In[23]:


model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)


# In[24]:


y_pred = model.predict_proba(X_val)[:, 1]


# In[25]:


target_decision = y_pred>=0.5


# In[26]:


target_decision


# In[27]:


(y_val == target_decision).mean()


# In[28]:


df_pred = pd.DataFrame()


# In[29]:


df_pred['probability'] = y_pred


# In[30]:


df_pred['prediction'] = target_decision.astype(int)


# In[31]:


df_pred['actual'] = y_val


# In[32]:


df_pred['correct'] = df_pred.prediction == df_pred.actual


# In[33]:


df_pred


# In[34]:


df_pred.correct.mean()


# In[35]:


from sklearn.metrics import accuracy_score


# In[36]:


accuracy_score(y_val, y_pred>=0.5)


# Confusion table

# In[37]:


y_val.sum()


# In[38]:


actual_positive = (y_val ==1)
actual_negative = (y_val == 0)
t = 0.4
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)


# In[39]:


tp = (actual_positive & predict_positive).sum()
tn = (actual_negative & predict_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# In[40]:


confusion_matrix = np.array([
    [tn,fp],
    [fn,tp]
])


# In[41]:


confusion_matrix


# In[42]:


confusion_matrix.sum()


# In[43]:


(confusion_matrix/confusion_matrix.sum()).round(3)


# # Precision and recall

# In[44]:


(tp+tn)/(tp+tn+fp+fn)


# In[45]:


p = tp/(tp + fp)
p


# Of all the people we predicted ho have heart deaseas, only 82% actuall hade it.

# In[46]:


r = tp/(tp + fn)
r


# we identified 85% people who actually had heart deaease, we missed 15%

# # ROC curve

# In[47]:


tpr = tp/(tp + fn)
fpr = fp/(fp + tn)


# In[48]:


tpr, fpr


# In[49]:


scores = []

thresholds = np.linspace(0,1,101)

for t in thresholds:
  actual_positive = (y_val ==1)
  actual_negative = (y_val == 0)

  predict_positive = (y_pred >= t)
  predict_negative = (y_pred < t)

  fp = (predict_positive & actual_negative).sum()
  fn = (predict_negative & actual_positive).sum()

  tp = (predict_positive & actual_positive).sum()
  tn = (predict_negative & actual_negative).sum()

  scores.append((t,tp,fp,fn, tn))


# In[50]:


columns = ['threshold','tp','fp','fn','tn']
df_scores = pd.DataFrame(scores, columns = columns)

df_scores['tpr'] = df_scores.tp/(df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp/(df_scores.fp + df_scores.tn)


# In[51]:


df_scores[::10]


# In[52]:


plt.plot(df_scores.threshold, df_scores.tpr, label= 'TPR')
plt.plot(df_scores.threshold, df_scores.fpr, label = 'FPR')
plt.legend()


# # Random model

# In[53]:


np.random.seed(1)
y_rand = np.random.uniform(0,1, size = len(y_val))


# In[54]:


y_rand.round(3)


# In[55]:


((y_rand >= 0.4) == y_val).mean()


# In[56]:


def tpr_fpr_dataframe(y_val, y_pred):
  scores = []

  thresholds = np.linspace(0,1,101)

  for t in thresholds:
    actual_positive = (y_val ==1)
    actual_negative = (y_val == 0)

    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    scores.append((t,tp,fp,fn, tn))

  columns = ['threshold','tp','fp','fn','tn']
  df_scores = pd.DataFrame(scores, columns = columns)

  df_scores['tpr'] = df_scores.tp/(df_scores.tp + df_scores.fn)
  df_scores['fpr'] = df_scores.fp/(df_scores.fp + df_scores.tn)

  return df_scores


# In[57]:


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# In[58]:


df_rand[::10]


# In[59]:


plt.plot(df_rand.threshold, df_rand.tpr, label = 'TPR')
plt.plot(df_rand.threshold, df_rand.fpr, label = 'FPR')
plt.legend()


# # ideal model

# In[60]:


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()

num_neg, num_pos


# In[61]:


y_ideal = np.repeat([0,1], [num_neg, num_pos])
y_ideal


# In[62]:


y_ideal_pred = np.linspace(0,1, len(y_val))


# In[63]:


1-y_val.mean()


# In[64]:


((y_ideal_pred >= 0.42934782608695654) == y_ideal).mean()


# In[65]:


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)


# In[66]:


df_ideal[::10]


# In[67]:


plt.plot(df_ideal.threshold, df_ideal.tpr, label = 'TPR')
plt.plot(df_ideal.threshold, df_ideal.fpr, label = 'FPR')
plt.legend()

plt.plot(df_rand.threshold, df_rand.tpr, label = 'TPR')
plt.plot(df_rand.threshold, df_rand.fpr, label = 'FPR')
plt.legend()

plt.plot(df_scores.threshold, df_scores.tpr, label= 'TPR')
plt.plot(df_scores.threshold, df_scores.fpr, label = 'FPR')
plt.legend()


# In[68]:


plt.figure(figsize = (5,5))

plt.plot(df_scores.fpr, df_scores.tpr, label = 'model')
plt.plot([0,1],[0,1],  label = 'random')
#plt.plot(df_rand.fpr, df_rand.tpr, label = 'random')
plt.plot(df_ideal.fpr, df_ideal.tpr,  label = 'ideal')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[69]:


from sklearn.metrics import roc_curve


# In[70]:


y_pred


# In[71]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[72]:


plt.figure(figsize = (5,5))

plt.plot(fpr, tpr, label = 'model')
plt.plot([0,1],[0,1],  label = 'random')


plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[73]:


plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)

plt.show()


# In[74]:


from sklearn.metrics import auc


# In[75]:


auc(df_scores.fpr, df_scores.tpr)


# In[76]:


auc(fpr, tpr)


# In[77]:


auc(df_ideal.fpr, df_ideal.tpr)


# In[78]:


def train(df,y_train, C = 1):
  dicts = df[categorical + numerical].to_dict(orient = 'records')

  dv = DictVectorizer(sparse = False)

  X_train = dv.fit_transform(dicts)

  model = LogisticRegression(C=C, max_iter= 100000)
  model.fit(X_train,y_train)

  return dv, model


# In[79]:


dv, model = train(df_train,  y_train, C=0.001)


# In[80]:


def predict(df, dv, model):
  dicts = df[categorical + numerical].to_dict(orient = 'records')
  X = dv.transform(dicts)

  y_pred = model.predict_proba(X)[:, 1]

  return y_pred


# In[81]:


y_pred = predict(df_val, dv, model)


# In[82]:


from sklearn.model_selection import KFold


# In[83]:


kfold = KFold(n_splits=10, shuffle=True, random_state=1)


# In[84]:


next(kfold.split(df_full_train))


# In[85]:


train_idx, val_idx = next(kfold.split(df_full_train))


# In[86]:


len(train_idx), len(val_idx)


# In[87]:


df_full_train.iloc[train_idx]


# In[88]:


df_train = df_full_train.iloc[train_idx]
df_val = df_full_train.iloc[val_idx]


# In[89]:


from tqdm.auto import tqdm


# In[90]:


from sklearn.metrics import roc_auc_score


# In[91]:


n_splits = 5
for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
  kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

  scores = []

  for train_idx, val_idx  in  kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.target.values
    y_val = df_val.target.values

    dv, model = train(df_train,  y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
  print(f'C={C} {np.mean(scores).round(3)}, {np.std(scores).round(3)}')


# In[92]:


scores


# In[93]:


np.mean(scores).round(3), np.std(scores).round(3)


# In[94]:


dv, model = train(df_full_train, df_full_train.target.values, C=0.1)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# Load Model

# In[95]:


import pickle


# In[96]:


output_file = f'model_C={C}.bin'
output_file


# In[97]:


'''f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()'''


# In[98]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)


# Load model

# In[1]:


import pickle


# In[2]:


model_file = 'model_C=10.bin'


# In[3]:


with open(model_file, 'rb') as f_in:
    (dv,model) = pickle.load(f_in)


# In[4]:


dv, model


# In[5]:


patient = {
    'sex': 'Female',
    'cp': 'atypical angina',
    'fbs': 'false',
    'restecg': 'normal',
    'exang': 'false',
    'slope': 'upsloping',
    'thal': 'normal',
    'age': 71,
    'trestbps': 160.0,
    'chol': 302.0,
    'thalch': 162.0,
    'oldpeak': 0.4,
    'ca': 2.0
}


# In[6]:


hasattr(dv, "vocabulary_"), hasattr(dv, "feature_names_")


# In[7]:


X_patient = dv.transform([patient])
prob = model.predict_proba(X_patient)[0, 1]
prob


# In[ ]:




