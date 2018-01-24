

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv('/home/shubhankar/Documents/csv/ex1data1.txt',sep = ",",header = None)
```


```python
art = np.random.uniform(size=(10,3))

```


```python
type(art)
```




    numpy.ndarray




```python
data.columns = ['x','y']

```


```python
data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
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
      <td>17.59200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.5277</td>
      <td>9.13020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.5186</td>
      <td>13.66200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.0032</td>
      <td>11.85400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.8598</td>
      <td>6.82330</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.3829</td>
      <td>11.88600</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.4764</td>
      <td>4.34830</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.5781</td>
      <td>12.00000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.4862</td>
      <td>6.59870</td>
    </tr>
    <tr>
      <th>9</th>
      <td>5.0546</td>
      <td>3.81660</td>
    </tr>
    <tr>
      <th>10</th>
      <td>5.7107</td>
      <td>3.25220</td>
    </tr>
    <tr>
      <th>11</th>
      <td>14.1640</td>
      <td>15.50500</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5.7340</td>
      <td>3.15510</td>
    </tr>
    <tr>
      <th>13</th>
      <td>8.4084</td>
      <td>7.22580</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5.6407</td>
      <td>0.71618</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.3794</td>
      <td>3.51290</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6.3654</td>
      <td>5.30480</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5.1301</td>
      <td>0.56077</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6.4296</td>
      <td>3.65180</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7.0708</td>
      <td>5.38930</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6.1891</td>
      <td>3.13860</td>
    </tr>
    <tr>
      <th>21</th>
      <td>20.2700</td>
      <td>21.76700</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5.4901</td>
      <td>4.26300</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6.3261</td>
      <td>5.18750</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5.5649</td>
      <td>3.08250</td>
    </tr>
    <tr>
      <th>25</th>
      <td>18.9450</td>
      <td>22.63800</td>
    </tr>
    <tr>
      <th>26</th>
      <td>12.8280</td>
      <td>13.50100</td>
    </tr>
    <tr>
      <th>27</th>
      <td>10.9570</td>
      <td>7.04670</td>
    </tr>
    <tr>
      <th>28</th>
      <td>13.1760</td>
      <td>14.69200</td>
    </tr>
    <tr>
      <th>29</th>
      <td>22.2030</td>
      <td>24.14700</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>10.2360</td>
      <td>7.77540</td>
    </tr>
    <tr>
      <th>68</th>
      <td>5.4994</td>
      <td>1.01730</td>
    </tr>
    <tr>
      <th>69</th>
      <td>20.3410</td>
      <td>20.99200</td>
    </tr>
    <tr>
      <th>70</th>
      <td>10.1360</td>
      <td>6.67990</td>
    </tr>
    <tr>
      <th>71</th>
      <td>7.3345</td>
      <td>4.02590</td>
    </tr>
    <tr>
      <th>72</th>
      <td>6.0062</td>
      <td>1.27840</td>
    </tr>
    <tr>
      <th>73</th>
      <td>7.2259</td>
      <td>3.34110</td>
    </tr>
    <tr>
      <th>74</th>
      <td>5.0269</td>
      <td>-2.68070</td>
    </tr>
    <tr>
      <th>75</th>
      <td>6.5479</td>
      <td>0.29678</td>
    </tr>
    <tr>
      <th>76</th>
      <td>7.5386</td>
      <td>3.88450</td>
    </tr>
    <tr>
      <th>77</th>
      <td>5.0365</td>
      <td>5.70140</td>
    </tr>
    <tr>
      <th>78</th>
      <td>10.2740</td>
      <td>6.75260</td>
    </tr>
    <tr>
      <th>79</th>
      <td>5.1077</td>
      <td>2.05760</td>
    </tr>
    <tr>
      <th>80</th>
      <td>5.7292</td>
      <td>0.47953</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.1884</td>
      <td>0.20421</td>
    </tr>
    <tr>
      <th>82</th>
      <td>6.3557</td>
      <td>0.67861</td>
    </tr>
    <tr>
      <th>83</th>
      <td>9.7687</td>
      <td>7.54350</td>
    </tr>
    <tr>
      <th>84</th>
      <td>6.5159</td>
      <td>5.34360</td>
    </tr>
    <tr>
      <th>85</th>
      <td>8.5172</td>
      <td>4.24150</td>
    </tr>
    <tr>
      <th>86</th>
      <td>9.1802</td>
      <td>6.79810</td>
    </tr>
    <tr>
      <th>87</th>
      <td>6.0020</td>
      <td>0.92695</td>
    </tr>
    <tr>
      <th>88</th>
      <td>5.5204</td>
      <td>0.15200</td>
    </tr>
    <tr>
      <th>89</th>
      <td>5.0594</td>
      <td>2.82140</td>
    </tr>
    <tr>
      <th>90</th>
      <td>5.7077</td>
      <td>1.84510</td>
    </tr>
    <tr>
      <th>91</th>
      <td>7.6366</td>
      <td>4.29590</td>
    </tr>
    <tr>
      <th>92</th>
      <td>5.8707</td>
      <td>7.20290</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5.3054</td>
      <td>1.98690</td>
    </tr>
    <tr>
      <th>94</th>
      <td>8.2934</td>
      <td>0.14454</td>
    </tr>
    <tr>
      <th>95</th>
      <td>13.3940</td>
      <td>9.05510</td>
    </tr>
    <tr>
      <th>96</th>
      <td>5.4369</td>
      <td>0.61705</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 2 columns</p>
</div>




```python
## lets plot the data
plt.scatter(data['x'],data['y'])
plt.show()
```


![png](output_6_0.png)


Our hypothesis is the following equation
h(x) = B0 + B1.x 

h(x) = B0.x0 + B1.x       (Where x0 = 1)

h (x) = [B0 + B1].[x0 + x1]



```python
x0 = [1]*len(x)
x1 = x.tolist()
```


```python
X = np.matrix([x0, x1])
```


```python
x = data.iloc[:,0]
y = data.iloc[:,1]
m = len(x)
iteration = 1500
alpha = 0.000005
theta_0 = 10
theta_1 = 0
# hypo(i) = theta_0 + (theta_1*x(i)) 
# cost_func J = 
```


```python
X = np.array([x0, x1])
```


```python
x0 = pd.Series([1]*len(x))
```


```python
def cost_fun_cal(para):
    cost_func = 0
    for i in range (len(x)):
        cost_func =  cost_func + (((para*x[i]- y[i])**2)/2*len(x))
    return (cost_func)

```


```python
li = []
starter = 10
for i in range(1,2000):
    starter = starter - 0.01
    li.append(starter)
```


```python
cost_li = []
for i in range(2000-1):
    cost_aart = cost_fun_cal(li[i])
    cost_li.append(cost_aart)
```


```python
plt.plot(li,cost_li)
plt.show()
```


![png](output_16_0.png)



```python
li[cost_li.index(min(cost_li))]
```




    0.8000000000001695




```python
theta_0_list = []
theta_0 = 10
for i in range(2000):
    temp_cost = cost_fun_cal(theta_0)
    #print (temp_cost,theta_0)
    theta_0 = theta_0 - alpha*(temp_cost)
    cost_func_list.append(temp_cost)
    theta_0_list.append(theta_0)

#plt.plot(cost_func_list)
#plt.show()
#print (cost_func_list)
#print (theta_0_list)
```

    /home/shubhankar/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:5: RuntimeWarning: overflow encountered in double_scalars
      """



```python
temcost = cost_fun_cal(10)
print (temcost)
teta_0 = 10
a = 10e-7
teta_0 = teta_0 - (a*temcost)
teta_0
```

    32451465.7811





    -152.2573289056628




```python
cost_func_list = []
theta_0 = []
for i in range(10):
    cost_func_list.append(i)

```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

