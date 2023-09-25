import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
df = pd.read_csv("test_scores.csv")
df.columns

x = np.array(df.math)
y = np.array(df.cs)

#calculate gradient descent
def gradient_descent(x, y):
    m = 0
    b = 0
    iteration = 1000000
    learning_rate = 0.0002
    l = len(x)
    cost_previous = 0
    
    for i in range(iteration):
        y_predict = m * x + b
        cost = (1/l) * sum([j**2 for j in (y - y_predict)])
         
        md = -(2/l) * sum(x*(y - y_predict))
        bd = -(2/l) * sum(y - y_predict)
        
        m = m - learning_rate * md
        b = b - learning_rate * bd
        
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        
        print("m = {}, b = {}, cost = {}, iteration = {}".format(m, b, cost, i))
    
    return m, b

m_rad, b_rad = gradient_descent(x, y)

#find m and b using linear regression model
reg2 = LinearRegression()
reg2.fit(df[['math']], df.cs)
print(reg2.coef_, reg2.intercept_) 

print(m_rad, b_rad) #m and b find using gradient descent function
