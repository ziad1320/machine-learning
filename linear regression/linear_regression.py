import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_set = pd.read_csv('Salary_Data.csv')
X_train = training_set['YearsExperience'].values
y_train = training_set['Salary'].values
# print(X_train,y_train)

def cost_function(x,y,w,b):
    m = len(x)
    cost_sum = 0
    f_wb = w*x + b
    for i in range(m):
        cost_sum += (f_wb[i] - y[i]) ** 2
    total_cost = (1/(2*m)) * cost_sum
    return total_cost

def gradient_function(x,y,w,b):
    m = len(x)
    dj_dw = 0
    dj_db = 0
    f_wb = w*x + b
    for i in range(m):
        dj_dw += (f_wb[i] - y[i]) * x[i]
        dj_db += (f_wb[i] - y[i])
    dj_dw = (1/m) * dj_dw
    dj_db = (1/m) * dj_db
    return dj_dw,dj_db      

def gradient_descent(x,y,alpha,iter):
    w = 0
    b = 0
    for i in range(iter):
        dj_dw,dj_db = gradient_function(x,y,w,b)
        w -= alpha * dj_dw
        b -= alpha * dj_db  
        if i % 50 == 0:
            print(f'iter {i}: cost {cost_function(x,y,w,b)}')
    return w,b

learning_rate = 0.01
iterations = 10000
final_w,final_b = gradient_descent(X_train,y_train,learning_rate,iterations)
print(f'w: {final_w:.4f}, b:{final_b:.4f}')

plt.scatter(X_train,y_train,color='blue',label='data points')
x_vals = np.linspace(min(X_train), max(X_train), 100)
y_vals = final_w * x_vals + final_b
plt.plot(x_vals, y_vals, color='red', label='Regression Line')
plt.xlabel('years of experince')
plt.ylabel('salary')
plt.legend()
plt.show()
