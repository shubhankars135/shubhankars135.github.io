---
layout: post
title:  Application of SVM's on Imbalanced (Skewed) Data
---

Dealing with unbalanced data in Machine learning has always been an area of confusion and applying complex algorithms on them make's the zone more difficult to dive in. The classification problem we will deal with is binary in nature and has less no. of variables as compared with a typical SVM case, but lets see if SVM works here. Lets get our hands dirty


```python
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```


```python
data = pd.read_csv('/home/shubhankar/Documents/csv/creditcard.csv')
data.head()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



Checking the imbalance or the level of skewness


```python
class_0_len =len(data[data['Class']==0])
class_1_len =len(data[data['Class']==1])

class_count = pd.DataFrame([class_1_len,class_0_len])
class_count.plot(kind = 'bar')
plt.show()

print (class_1_len/len(data),class_0_len/len(data))
```


![png](output_6_0.png)


    0.001727485630620034 0.9982725143693799


Surely the dataset is very imbalanced, so it has to be treated in quite a different manner. Following is our plan of activites to successfully apply SVM's on this.

### 1 . Data Preprocessing : 

    -> Get rid of 'Time' column 
    -> Scale the 'Normal' column  
    -> Removal of outliers: SVM's work well only when there is no outlier, so lets get rid of them before modelling


```python
from sklearn.preprocessing import StandardScaler
data['amount_norm'] = StandardScaler().fit_transform(data['Amount'].reshape(-1,1))
data = data.drop(['Amount','Time'],axis = 1)
```


```python
## separating the predictors and response columns
X = data.drop(['Class'],axis = 1)
y = data['Class']
```

Now we've got rid of 'Time' column and scaled the 'Amount' column, now lets remove the outliers. 


```python
## lets check the possible outliers in 'V1' , 'V2', 'V3' , 'V4'
percentiles  = [5, 50, 90, 95, 96, 97, 98, 99, 100]
print (stats.scoreatpercentile(X['V1'], percentiles)) 
print (stats.scoreatpercentile(X['V2'], percentiles)) 
print (stats.scoreatpercentile(X['V3'], percentiles))
print (stats.scoreatpercentile(X['V4'], percentiles))
```

    [-2.89914677  0.0181088   2.01540883  2.08122306  2.10288811  2.12665607
      2.16656766  2.23712971  2.45492999]
    [ -1.97197514   0.06548556   1.32663471   1.80858475   1.99343663
       2.26459168   2.7581922    3.80181119  22.05772899]
    [-2.38974046  0.17984634  1.67617301  2.06263514  2.17282221  2.30873932
      2.47280427  2.72843414  9.38255843]
    [ -2.19568282  -0.01984653   1.48280662   2.56650066   2.81700044
       3.20051629   3.72089022   4.24803168  16.87534403]


As visible there are some outliers in V2, V3 and V4. Now, we can't check manually if there are outliers in all 29 predictors so lets write a loop, such that if the 100 percentile is greater than 3 times of the 99 percentile then replace all values above 99 percentile with 99 percentile values itself.  


```python
col_names = list(X) ## list of all predictor column names 
affected_columns = []  ## lets see for how many columns have outilers

for i in col_names:
    if stats.scoreatpercentile(X[i], 100) > 3*(stats.scoreatpercentile(X[i], 99)):
        X.loc[X[i] > stats.scoreatpercentile(X[i], 99), i] = stats.scoreatpercentile(X[i], 99)
        affected_columns.append(i)
    else:
        pass
    
print(len(np.array(affected_columns)),np.array(affected_columns))      
```

    25 ['V2' 'V3' 'V4' 'V5' 'V6' 'V7' 'V8' 'V9' 'V10' 'V11' 'V12' 'V14' 'V15'
     'V16' 'V17' 'V20' 'V21' 'V22' 'V23' 'V24' 'V25' 'V26' 'V27' 'V28'
     'amount_norm']


25 columns had outliers in them, which were succesfully replaced. Now hopefully the data is ready for modelling. 


```python
## Adding the X and y to form a new dataset ,remember the original dataset(data) still has outliers
clean_data = pd.concat([X,y], axis = 1)
```

### 2. Modelling
We are going fit the SVM's on two datasets, which will be

    -> undersampled dataset (We won't be performing oversampling for two reasons : it just adds same observations to our dataset which will probably cause a lot of overfitting with SVM, and secondly since it'll increase the size of dataset it would be quite difficult computionally to fit SVM on it)
    -> the original dataset as we want to compare the results of SVM on both.
    -> We can also test our SVM on artificial data generated by a method called SMOTE.
    
#### Performing undersampling : 

Undersampling : In a skewed (imbalanced) dataset we just randomly select n number of observations from the majority class (n = no. of obs. from minority class) and dump the others. So basically we are just reducing the no. of observation in majority class and make them equal to no. of obs. from minority class, so that we get a balanced dataset. 


```python
# picking up the indices
major_class_index = clean_data[clean_data['Class']==0].index
minor_class_index = clean_data[clean_data['Class']==1].index

# randomly picking 'class_1_len' no. of indices of from major_class list
random_major_index = np.random.choice(major_class_index, class_1_len, replace = False)
random_major_index = np.array(random_major_index)

# adding the list of indices
under_sample_index = np.concatenate([minor_class_index, random_major_index])

under_sample_data = clean_data.iloc[under_sample_index,:]
```


```python
# now u can see both the classes are of same length
print (len(under_sample_data[under_sample_data['Class']==1])/len(under_sample_data),
       len(under_sample_data[under_sample_data['Class']==0])/len(under_sample_data))
```

    0.5 0.5



```python
# separating X and y in under_sample_data

X_under = under_sample_data.iloc[:,under_sample_data.columns !='Class']
y_under = under_sample_data['Class']

```


```python
## splitting both datasets into test and train

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0 )
X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X_under,y_under,test_size = 0.3,random_state = 0)

```

## Fitting SVM

### Approach
1.)Firstly we will fit a SVM with linear kernel on under sampled data. To select the optimal C value we will do cross valdation. We will consider the average recall_scores to do so. To know more about precision and recall , visit [this documentation](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

2.) Then we will check model(with the optimal C value) this on undersampled data

3.) Now we will fit the model on the entire dataset(not undersampled). And see if there's a difference.

4.)Then we will perform the above steps with SVM with radial-basis-function kernel (RBF). There we have to choose two values - C value and Gamma. To know more about the significance of this values visit [this documentation](http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)

Since we are more concerned about classifying the Frauds(class = 1) correctly, we will be more focused on the recall the score, since it captures this very well.
#### Remember 
>Recall = TP / (TP + FN)              where TP - Frauds which were correctly classified, FN - Frauds wrongly classified

>Accuracy = TP + TN/ (Total no. of obs.)         where TN - Non-Frauds which are correctly classified.

So the no. of TN will be very high compared to TP, as the no. of obs. in class 0 is very high overall, so accuracy wont be able to judge how good our model is performing.





```python
from sklearn import svm
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score
```

Let us write a function will perform cross validation and test different C values on the linear kernel SVM, let us also add a feature which will calculate the average recall and accuracy for each iterated datasets given by cross validation 


```python
def cross_validation (X_data, y_data):
    folds_list = KFold(len(X_data),5,shuffle = True)
    
    C_values = [0.001, 0.01, 0.1,0.2,0.3,0.4, 1,10 ,100]
    

    for c_value in C_values:
        print ("_______________ c value____________:{0}".format(c_value))
        print ("  Recall acc    Accuracy")
        recall_accuracies = []
        accuracy_values = []
    
        for iterations,indices in enumerate(folds_list,start = 1):
            # indices[0] will consist of training sets, indices[1] of test set

            # fitting svm on train indices
            clf = svm.SVC(kernel = 'linear', C = c_value)
            y_under_array = np.array(y_data.iloc[indices[0],])
            clf.fit(X_data.iloc[indices[0],:],y_under_array)
            

            # predicting on test indices
            y_pred = clf.predict(X_data.iloc[indices[1],:].values)

            # checking the recall_values
            recall_value = recall_score(y_data.iloc[indices[1],].values,y_pred)
            recall_accuracies.append(recall_value)
            
            # checking the accuracy of the model
            accuracy = accuracy_score(y_data.iloc[indices[1],].values,y_pred)
            accuracy_values.append(accuracy)

            ## printing the results 
            print ("%d  %.5f      %.5f " %(iterations,recall_value, accuracy))
            
        ## print the avg. recall accuracy for a c value
        print (" ")
        print ("avg recall accuracy : {0}".format(np.array(recall_accuracies).mean()))
        print ("avg accuracy        : {0}".format(np.array(accuracy_values).mean()))
        print (' ')
```


```python
cross_validation(X_under,y_under)  
```

    _______________ c value____________:0.001
      Recall acc    Accuracy
    1  0.84706      0.93401 
    2  0.85437      0.92386 
    3  0.80769      0.89848 
    4  0.85455      0.91878 
    5  0.82222      0.91837 
     
    avg recall accuracy : 0.8371775480056461
    avg accuracy        : 0.918698850098415
     
    _______________ c value____________:0.01
      Recall acc    Accuracy
    1  0.85882      0.91878 
    2  0.89320      0.93401 
    3  0.85577      0.92386 
    4  0.90000      0.93909 
    5  0.90000      0.94388 
     
    avg recall accuracy : 0.8815593287352282
    avg accuracy        : 0.9319227183259091
     
    _______________ c value____________:0.1
      Recall acc    Accuracy
    1  0.94118      0.94924 
    2  0.90291      0.94924 
    3  0.87500      0.92893 
    4  0.90909      0.94924 
    5  0.90000      0.94388 
     
    avg recall accuracy : 0.9056360002076737
    avg accuracy        : 0.9441054594426603
     
    _______________ c value____________:0.2
      Recall acc    Accuracy
    1  0.94118      0.94924 
    2  0.91262      0.94924 
    3  0.87500      0.92893 
    4  0.90909      0.94924 
    5  0.90000      0.93878 
     
    avg recall accuracy : 0.9075777477804892
    avg accuracy        : 0.9430850512793951
     
    _______________ c value____________:0.3
      Recall acc    Accuracy
    1  0.94118      0.94924 
    2  0.91262      0.94924 
    3  0.87500      0.92893 
    4  0.90909      0.94924 
    5  0.91111      0.94388 
     
    avg recall accuracy : 0.9097999700027113
    avg accuracy        : 0.9441054594426603
     
    _______________ c value____________:0.4
      Recall acc    Accuracy
    1  0.94118      0.94416 
    2  0.91262      0.94924 
    3  0.87500      0.92893 
    4  0.90909      0.94924 
    5  0.91111      0.94388 
     
    avg recall accuracy : 0.9097999700027113
    avg accuracy        : 0.9430902310162643
     
    _______________ c value____________:1
      Recall acc    Accuracy
    1  0.94118      0.94416 
    2  0.91262      0.94416 
    3  0.87500      0.92386 
    4  0.91818      0.95431 
    5  0.93333      0.95408 
     
    avg recall accuracy : 0.9160625962653375
    avg accuracy        : 0.9441158189163991
     
    _______________ c value____________:10
      Recall acc    Accuracy
    1  0.94118      0.94416 
    2  0.91262      0.93909 
    3  0.87500      0.91878 
    4  0.91818      0.94924 
    5  0.93333      0.94898 
     
    avg recall accuracy : 0.9160625962653375
    avg accuracy        : 0.9400497254739459
     
    _______________ c value____________:100
      Recall acc    Accuracy
    1  0.94118      0.93909 
    2  0.90291      0.93401 
    3  0.87500      0.91878 
    4  0.91818      0.95431 
    5  0.94444      0.95408 
     
    avg recall accuracy : 0.9163430709147443
    avg accuracy        : 0.9400549052108153
     


Depending upon the Random test and train set given by Cross validation the recall and accuracy will vary.The accuracy doesnt seem to change much but the recall value changes with the c parameter.A bit of observation and after running the loop many times i've found that C = 0.2 gives the best tradeoff between complexity (greater C value) and recall metric.



```python
## applying linear svm to undersampled data with the optimal C parameter
clf_1 = svm.SVC(kernel = 'linear', C = 0.2)
y_under_array = np.array(y_train_under,)
clf_1.fit(X_train_under,y_under_array)

## predicting on x_under_test and calculating recall on y_under_test
y_under_pred = clf_1.predict(X_test_under)

## calculating the recall score
recall_clf_1 = recall_score(y_test_under,y_under_pred)
accu_1 = accuracy_score(y_test_under,y_under_pred)
print ('recall accuracy : {0} '.format(float(recall_clf_1)))
print ('accuracy        : {0}'.format(float(accu_1)))
```

    recall accuracy : 0.9047619047619048 
    accuracy        : 0.9358108108108109


Lets be content with these results and apply this model to the entire dataset(which was not undersampled).


```python
## applying linear svm to entire data with the optimal C parameter
clf_2 = svm.SVC(kernel = 'linear', C = 0.2)
y_under_array = np.array(y_train,)
clf_2.fit(X_train,y_under_array)

## predicting on x_under_test and calculating recall on y_under_test
y_pred = clf_2.predict(X_test)

## calculating the recall score
recall_clf_2 = recall_score(y_test,y_pred)
accu_2 = accuracy_score(y_test,y_under_pred)
print ('recall accuracy : {0} '.format(float(recall_clf_2)))
print ('accuracy        : {0}'.format(float(accu_2)))
```

    recall accuracy : 0.7920792079207921 
    accuracy        : 0.999385555282469


Thus undersampling surely helps !!!!!!!!!! We can see that there is quite a difference between recall score of undersampled data (0.904) and of non-undersampled data (0.792)
One intresting thing to try is trying this SMOTE generated data.

## Fitting the model on SMOTE data

   This is the good way to tackle imbalanced data problem. It might be useful to know the SMOTE(Synthetic Minority Oversampling Technique) adds synthetically generated artificial data to minority class, in this way it does'nt replicate data (as done in random oversampling) thus preventing overfitting 
   
   Now a mistake one can perform here is perform SMOTE operation on overall dataset and then split it into train and test.SMOTE is only to be applied on train dataset and not on test dataset. Why??? . Nice question. A quick answer is although SMOTE doesnt replicate values, it uses the nearest neighbors of observations to create synthetic data, thus we can say it generates similar values .
   
   So if we perform SMOTE on entire dataset, some information (or variation) from test dataset might leak into train dataset, as the train test split is performed at later stage. And hence, a complex model(like SVM) will be able to perfectly predict the value for those observations when predicting for the test set,making accuracy and recall, dubious.
   Enough talk!!, lets see it practically



```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(X_train, y_train)

```

    /home/shubhankar/anaconda3/lib/python3.6/site-packages/sklearn/utils/deprecation.py:77: DeprecationWarning: Function _ratio_float is deprecated; Use a float for 'ratio' is deprecated from version 0.2. The support will be removed in 0.4. Use a dict, str, or a callable instead.
      warnings.warn(msg, category=DeprecationWarning)



```python
print ('Class_1 lenght = {0}'.format(len(y_train_res[y_train_res==1])))
print ('Class_0 lenght = {0}'.format(len(y_train_res[y_train_res==0])))
```

    Class_1 lenght = 227454
    Class_0 lenght = 227454


So SMOTE has done its job and has balanced the data, lets fit the model on it


```python
## applying linear svm to SMOTE data with the optimal C parameter
clf_2 = svm.SVC(kernel = 'linear', C = 0.2)
y_res = np.array(y_train_res,)
clf_2.fit(x_train_res,y_res)

## predicting on x_under_test and calculating recall on y_under_test
y_under_pred2 = clf_2.predict(X_test)

## calculating the recall score
recall_clf2 = recall_score(y_test,y_under_pred2)
accu2 = accuracy_score(y_test,y_under_pred2)
print ('recall accuracy : {0} '.format(float(recall_clf2)))
print ('accuracy        : {0}'.format(float(accu2)))
```

    recall accuracy : 0.9306930693069307 
    accuracy        : 0.9811979916435518


Much better!!! The recall has jumped up from 90.4 (achieved by undersampled data) to 93.06 (achieved by using the SMOTE data)

## Conclusion 
Applying SVM on an Imbalanced data without any preprocessing does'nt give a good performance .We saw that results are much better with undersampled data and even better with SMOTE generated data.
