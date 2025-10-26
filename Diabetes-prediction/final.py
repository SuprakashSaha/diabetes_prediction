import numpy as np
import pandas as pd
from sklearn import datasets
import joblib

diabetes = pd.read_csv("C:/Users/Admin/Desktop/Practice project/diabetis web app project/Diabetes-prediction/diabetes1.csv")


X=diabetes.data
y=diabetes.target

print(X.shape,y.shape)

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.3,random_state=99)
print(train_x.shape,train_y.shape)


from sklearn.linear_model import Ridge

le = Ridge(alpha=.1)
le.fit(train_x,train_y)

joblib.dump(le, 'dia.pkl')
print("Model trained and saved as 'model.pkl'")