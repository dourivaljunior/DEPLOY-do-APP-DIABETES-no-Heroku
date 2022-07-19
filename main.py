import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import pickle
from warnings import filterwarnings
filterwarnings("ignore")  ## To remove any kind of warning
data = pd.read_csv("diabetes.csv")
data.head()
#independent and dependent columns
x = data[['Pregnancies','Glucose','BloodPressure','SkinThickness',
          'Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = data['Outcome']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
logreg= LogisticRegression()
logreg.fit(x_train,y_train)
predictions=logreg.predict(x_test)
logreg.score(x_test,y_test)
score=logreg.score(x_test,y_test)
#save the model
file = open("logreg_model.pkl", 'wb')
pickle.dump(logreg, file)
