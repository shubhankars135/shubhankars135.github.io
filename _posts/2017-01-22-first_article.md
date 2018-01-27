---
layout: post
title:  Implementing Linear Regression by Gradient descent(from the scratch)
---
Implementing something from the scratch , as opposed to using a established libraries to arrive at the results, is what true ML is all about - Andrej Karpathy

Keeping the above in mind, my first article in this series is about implementing something as simple as linear regression from the scratch i.e without using scikit learn (which is the most widely used library to do that). We will briefly see what linear regreesion is (although this isn't a proper introduction to linear regression, but more about how to write your own functions to do so). Lets get started.

## Brief overview of Linear Regression

Firstly, for the better understanding of Gradient Descent,let us focus on Simple linear regression i.e. linear regression with only one predictor and one response involved. So to brief you roughly about linear regression. In linear regression we are trying to come up with a linear equation, which follows the varation of data in the closet possible manner. In more technical terms, we are finding the equation of a straight line, which has the least Sum of Squared Errors(SSE).This line is called the best fit line.

Lets say,our problem statement requires us to predict the sales of a product in various cities, given its population. So if x column represents the population of city in millons, and y represents the monthly sales in thousand USD, our aim is to come up with equation which looks something like

> Y = B0 + B1.X       .... (I)

Where

> Y = (y1,y2, ..... yn) &

> X = (X1,X2, ..... Xn)

Now our problem boils down to finding B0, B1

NOTE : This article is about gradient descent and its implementation and not about Linear regression. If the reader is unfamiliar to it, I strongly recommend going through this [article](https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv('/home/shubhankar/Documents/csv/ex1data1.txt',sep = ",",header = None)
data.columns = ['x','y']
x = data['x']
Y = data['y']
df.set_index('NAME', inplace=True)
data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.1101</td>
      <td>17.5920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.5277</td>
      <td>9.1302</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.5186</td>
      <td>13.6620</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.0032</td>
      <td>11.8540</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.8598</td>
      <td>6.8233</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.scatter(x, Y)
plt.xlabel('Population in millons')
plt.ylabel('Sales of product in thousand USD')
plt.show()
```


![_config.yml]({{ site.baseurl }}/images/scatter.png)


We will be defining X in the following manner, just following rules of linear algera


```python
x1 = np.array(x).reshape(97,1)
x0 = np.ones((97,1))
```


```python
X = np.hstack((x0,x1))
Y = Y.values.reshape(97,1)
```


```python
X = np.matrix(X)
Y = np.matrix(Y)
m = len(data)
print (X.shape, Y.shape)
```

    (97, 2) (97, 1)


Now **X**is 97 x 2 matrix, and **Y** is 97 x 1 matrix, lets define **B** = (B0,B1) (a 2 x 1 matrix) so that we could rewrite our (I)
equation as ...
> **Y = X . B**

## Cost function and its optimization.

  In general, most of the parametric ML algorithms have two major steps - there is cost function(or loss function) , and then its optimization (maximization or minimization) for it, which yields the optimal solution.
  
  In reality we can never find the above true $\theta$, but we can make a hypothesis,infact many hypotheses, and then choose the one which gives least error. That will be the optimum value for $\theta$

Let us define the cost function as:

![_config.yml]({{ site.baseurl }}/images/cost_function.png)


The minimization, or the local minimum of the above function will yield the optimum solution.The reason we divide it by 2, is because its mathematically easy to minimize the function. Our objective in gradient descent in context to the below image is to find point at the lowest height of plane.

![_config.yml]({{ site.baseurl }}/images/slr_grad.png)

#### We now define a function such that it takes B0, and B1 as parameters and returns the value of cost function. 

```python
def cost_cal(B_0,B_1):
    B = np.array([B_0,B_1]).reshape(2,1)
    B = np.matrix(B)
    h_i = X*B  ## our hypothesis
    diff = (np.square(h_i-Y))/(2*m)
    return (diff.sum())
```

## Gradient Descent.

   In gradient descent, we initialize our **B** to some random value, and then keep updating it until we've found the optimum solution, in other words untill convergence.
   Mathematically 
   
![_config.yml]({{ site.baseurl }}/images/Convergence.png)
   

Where a is the learning rate.

Now, two things,- how do we know that we've reached convergence. One easy way is to run the gradient descent algorithm multiple times and plot the value of cost function(corresponding to that B) against the no. of iterations. So if solve the partial differentiation term in the above equation. The algorithm now will be, something like. 

We will select the number os iterations and the no. of iterations randomly.Although I'm choosing them randomly there are some ways to concretly intiallize these parameters , but lets not dive into it.

#### The following function takes B0, B1 and learning rate  as parameters, performs a single step of gradient descent and returns the updated values of B0 and B1


```python
def gradient_descent(b_0,b_1,a):
    B_temp = np.array([b_0,b_1]).reshape(2,1)
    b_0 = b_0 - (a/m)*((X*B_temp - Y).sum())
    X_t = X.transpose()
    b_1 = b_1 - (a/m)*((X_t*(X*B_temp - Y)).sum())
    return (b_0,b_1)
```

Lets write a function which performs the gradient descent multiple times and returns a list of values of cost function, we will also print the values of updated B0, B1 we will also print the values of B0 and B1 to find the optimal solution.

>##### The optimal solution, i.e the optimal value of B will be that, for which the cost function has the minimum value.(local minimum)


```python
num_iters = 2500
```


```python
def multi_grad_descent(b0,b1,a):
    cost_fun_value_list = []
    for i in range(num_iters):
        if i == 0:
            result = gradient_descent(b0, b1, a)
            new_b0 = result[0]
            new_b1 = result[1]
            new_cost = cost_cal(new_b0, new_b1)
            cost_fun_value_list.append(new_cost)
        else:
            result = gradient_descent(new_b0,new_b1,a)
            new_b0 = result[0]
            new_b1 = result[1]
            new_cost = cost_cal(new_b0, new_b1)
            cost_fun_value_list.append(new_cost)
        print (new_b0,new_b1,new_cost)
        i += 1
    return(cost_fun_value_list)
```


```python
cost_list0 = multi_grad_descent(1,1,0.01)
## Note : I'm not priniting the values here because the limitation of printing 2500 values
```


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(list(range(num_iters)),cost_list0)
plt.show()
```

![_config.yml]({{ site.baseurl }}/images/grad.png)


As we can see from the above diagram, there isn't any significant desecnt taking place for the last 500 iterations.
So we can conclude that convergence has taken place.And as we can see from the printed results that 

> #### B0 = -3.8152 & B1 = 1.1847 correspond to minimum value of cost function, so its our optimum solution.

I encourage the reader to try a different set of input parameters and see the results.

Lets verfiy our results with that of scikit learns.


```python
from sklearn import linear_model
lr = linear_model.LinearRegression(fit_intercept = True)
model_1 = lr.fit(x.values.reshape(-1,1),Y)
print (" b0 = {0} ; b1 = {1}".format(model_1.intercept_,model_1.coef_[0]))
```

     b0 = [-3.89578088] ; b1 = [ 1.19303364]


Thus our values match the results with scikit learn's result. Further, there are also different types of gradient descent 
namely batch gradient descent and Stochastic gradient descent, details of which are beyond the scope of this article. 
Although this might be an [interesting read](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent)


