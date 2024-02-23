![Screenshot (5)](https://github.com/Ashokanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160997973/2821d3b2-b11f-461a-b407-7f40c5702f2d)![Screenshot (2)](https://github.com/Ashokanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160997973/4cb50884-b851-4cba-be50-9fcb096543f4)# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
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
![Screenshot (3)](https://github.com/Ashokanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160997973/5c5d7882-8120-49e5-b6ca-0d48f727b19d)
![Screenshot (2)](https://github.com/Ashokanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160997973/3521650c-b8de-42f9-9f1b-71acb03272d9)

![Screenshot (1)](https://github.com/Ashokanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160997973/b4fb0855-0638-44fb-873c-bacd26571358)
![Screenshot (5)](https://github.com/Ashokanan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160997973/e419a507-41bd-4063-90e2-b8305650f6c7)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
