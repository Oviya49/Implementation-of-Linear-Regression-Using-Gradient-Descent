# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.
```

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: Oviya.N
RegisterNumber: 212223040140 

```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
  X=np.c_[np.ones(len(X1)),X1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
     predictions = (X).dot(theta).reshape(-1,1)
     errors =(predictions-y).reshape(-1,1)
     theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
     return theta

data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())  

X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled= scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled, Y1_Scaled)
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value = {pre}")
```
## Output:
```
Dataset:
```
![image](https://github.com/user-attachments/assets/0a56ee13-815c-4646-a9fb-de173636dbd9)

```
X Values:
```
![image](https://github.com/user-attachments/assets/2e5ceefc-d2ba-4605-8a55-59fa261da367)

```
Y Values:
```
![image](https://github.com/user-attachments/assets/00221beb-1356-4b3e-9ef0-9dbd0d39b9cf)

```
Predicted value:
```
![image](https://github.com/user-attachments/assets/33adfa29-7a66-4dac-a981-07e554b2914c)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
