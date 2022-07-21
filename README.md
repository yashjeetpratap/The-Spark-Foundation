# The-Spark-Foundation
GRIP22 TASK 1
Task 1 :- Prediction using Supervised Machine Learning
Author -- YASHJEET PRATAP SINGH
Data scientist and Business analyst

# Importing the required libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
​
# Reading the data from the remote link
web= r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data= pd.read_csv(web)
print("Data import successful")
data.head(10)
Data import successful
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
Data processing
X=data.iloc[:, :-1].values
y=data.iloc[:, 1].values
Model training
y
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)
​
y
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
​
print("Training complete.")
Training complete.
Data Visualization
Plotting the line of regression
# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_
​
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()

Model testing
# Testing data - In Hours
print(X_test) 
# Predicting the scores
y_pred = regressor.predict(X_test) 
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]]
Comparing the Prediction model result with Actual result
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 
Actual	Predicted
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
df.plot(kind='bar',figsize=(5,5))
plt.grid(which='major', linewidth='0.5', color= 'pink')
plt.grid(which='minor', linewidth= '0.5', color= 'green')
plt.show
###  plotting the Bar the graph to depict the difference between the actual and prediction value
​
df.plot(kind='bar',figsize=(5,5))
plt.grid(which='major', linewidth='0.5', color= 'pink')
plt.grid(which='minor', linewidth= '0.5', color= 'green')
plt.show
<function matplotlib.pyplot.show(close=None, block=None)>

# Estimating training and test score
​
print("Training Score:",regressor.score(X_train,y_train))
print("Test Score:",regressor.score(X_test,y_test))
Training Score: 0.9515510725211552
Test Score: 0.9454906892105355
# we can also test with our own data
​
hours = 9.25
test= np.array([hours])
test= test.reshape(-1,1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
No of Hours = 9.25
Predicted Score = 93.69173248737535
Evaluating the model
:
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Square Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print('R-2:', metrics.r2_score(y_test,y_pred))
Mean Absolute Error: 4.183859899002975
Mean Square Error: 21.598769307217406
Root Mean Squared Error: 4.647447612100367
R-2: 0.9454906892105355
Conclusion
I've successfully able to carry out the Prediction using Supervised ML task and able to evaluate the model performance on various parameters

THANK YOU
