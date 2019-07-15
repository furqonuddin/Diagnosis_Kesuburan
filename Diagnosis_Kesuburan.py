import pandas as pd
import numpy as np

df = pd.read_csv('fertility.csv')

# print(df.isna().sum())
# print(df.dtypes)
df = df.drop(['Season'], axis='columns')
dfNew = pd.get_dummies(df[['Childish diseases', 'Accident or serious trauma',
                        'Surgical intervention', 'High fevers in the last year', 
                        'Frequency of alcohol consumption', 'Smoking habit'
                        ]])

df = pd.concat([df, dfNew], axis='columns')
df = df.drop([
            'Childish diseases', 'Accident or serious trauma',
            'Surgical intervention', 'High fevers in the last year', 
            'Frequency of alcohol consumption', 'Smoking habit'
            ], axis='columns')

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['Diagnosis'] = label.fit_transform(df['Diagnosis'])

x = df.drop(['Diagnosis'], axis='columns')

# Logistic_Regression
from sklearn.linear_model import LogisticRegression
modelLog = LogisticRegression(solver='liblinear')
modelLog.fit(x, df['Diagnosis'])
# print(modelLog.score(x, df['Diagnosis'])*100,'%')

# Decission_tree
from sklearn import tree
modelTree = tree.DecisionTreeClassifier()
modelTree.fit(x, df['Diagnosis'])
# print(modelTree.score(x, df['Diagnosis'])*100,'%')

# Random_forest
from sklearn.ensemble import RandomForestClassifier
modelForest = RandomForestClassifier(n_estimators=50)
modelForest.fit(x, df['Diagnosis'])
# print(modelForest.score(x, df['Diagnosis'])*100,'%')

# prediksi
dfProfil = pd.read_csv('profil.csv', delimiter=';')
pred = dfProfil.drop(['Nama'], axis='columns')

# Arin, prediksi kesuburan: NORMAL (Lasso Regression)
for i in (dfProfil.index.values):
    hl = modelLog.predict(pred.iloc[i].values.reshape(1,-1))[0]
    ht = modelTree.predict(pred.iloc[i].values.reshape(1,-1))[0]
    hf = modelForest.predict(pred.iloc[i].values.reshape(1,-1))[0]
    if hl==1:
        hl ='Normal'
    else:
        hl ='Not Normal'
    
    if ht==1:
        ht ='Normal'
    else:
        ht ='Not Normal'
    
    if hf==1:
        hf ='Normal'
    else:
        hf ='Not Normal'

    print(dfProfil['Nama'].iloc[i], ', Prediksi esuburan:', hl, '(', 'Logistic_Regression',')')
    print(dfProfil['Nama'].iloc[i], ', Prediksi esuburan:', ht, '(', 'Decission_tree',')')
    print(dfProfil['Nama'].iloc[i], ', Prediksi esuburan:', hf, '(', 'Random_forest',')')
    print('\n')
