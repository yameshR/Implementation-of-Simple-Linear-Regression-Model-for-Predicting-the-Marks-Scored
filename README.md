# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libaries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE,MAE and RMSE.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:Yamesh R
RegisterNumber:212222220059
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```
## Output:
![image](https://github.com/23004513/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138973069/0d06a575-94a7-40be-ad2f-b925a14db9ac)        
![image](https://github.com/23004513/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138973069/90a15f1a-c470-4482-b7d5-ca3ff99bc39e)      
![image](https://github.com/23004513/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138973069/2021ce29-6c1e-4511-ab0e-0e55787306dd)         
![image](https://github.com/23004513/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138973069/154f088d-3ca8-46ba-acb1-07dee94830d9)         
![image](https://github.com/23004513/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138973069/a3d1f641-5f76-49b3-9a53-7084d3867b54)       
![image](https://github.com/23004513/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138973069/d3b9639e-1e6d-498e-89d7-938f3650f95a)       


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.


