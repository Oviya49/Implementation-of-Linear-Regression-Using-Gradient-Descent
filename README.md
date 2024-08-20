# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Initialize parameters
2.Create feature matrix
3.Calculate predictions
4.Claculate erros
5.Update weights
6.Repeat
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
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):
    #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
    #calculate erros
        error=(predictions-y).reshape(-1,1)
    #update thera using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(error) 
    
    return theta

data=pd.read_csv("C:/Users/admin/Downloads/50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:
![image](https://github.com/user-attachments/assets/8890fe01-d272-4266-888f-1e922bcedb26)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
