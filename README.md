# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ADHITHIYAN .K
RegisterNumber:  212222230006
*/
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![275325718-0132c4fa-35e0-4b03-8383-52a06a5adba9](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/d7a0f1a5-28ab-4a44-a722-17c9eac703d7)

![275325749-79c50faa-26bf-4647-9540-2dd54615a67b](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/fbcbc57f-9cc2-4786-85d8-11affba029e4)

![275325786-cc1a8eea-2adf-4787-b698-341ac2a2e3a1](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/f66a1294-9ec4-4ded-92aa-8c62ade58445)

![275325835-33705919-842d-4475-b31e-c2a75f9a09c0](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/7c02cb28-1b07-454a-9f6f-bdbef77ab422)

![275325941-6ccdb4f3-4ccd-45c3-a9d5-1c2486b2f030](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/5240c436-694c-44d1-b5a6-e2c5a2315bf4)

![275325941-6ccdb4f3-4ccd-45c3-a9d5-1c2486b2f030](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/3bf6e8ad-cbb9-4158-8e35-8e47fe1617d4)

![275325979-7aca4148-7acf-4d39-a9d7-50164dff6bb8](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/46734310-d08d-4457-a42e-675859cdeaf7)

![275326025-0b5c5cd2-9127-4b08-aaa4-94e0e6a42197](https://github.com/AdhithiyanK/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121029258/60a1a9a0-f935-4312-9c7a-14060a4a330d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
