import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

db = datasets.load_diabetes()

#print(db.feature_names)

x_1 = pd.DataFrame(db.data, columns = db.feature_names)
#取age (年齡)、 sex (性別)、 bmi (Body Mass Index 體質指數)、bp (Average Blood Pressure 平均血壓)做訓練
x_2 = x_1.iloc[:,0:4]
y = pd.DataFrame(db.target, columns = ["Predicted Quantitatjve Measure"])

#建模
lm_1 = LinearRegression() 
lm_1.fit(x_1, y)
lm_2 = LinearRegression() 
lm_2.fit(x_2,y)

#畫圖
pre_tar1 = lm_1.predict(x_1) #根據x_1（所有資料），也就是用所有資料去預測
plt.scatter(y, pre_tar1)
plt.xlabel("Quantitative Measure")
plt.ylabel("Predicted Quantitive Measure")
plt.title("Quantitative Measure vs Predicted Quantitative Measure")
plt.show()

pre_tar2 = lm_2.predict(x_2) #根據x_2（age, sex, bmi, bp）預測
plt.scatter(y, pre_tar2)
plt.xlabel("Quantitative Measure")
plt.ylabel("Predicted Quantitive Measure")
plt.title("Quantitative Measure vs Predicted Quantitative Measure")
plt.show()

#----------------------------------------------------------------------

x = pd.DataFrame(db.data, columns = db.feature_names)
y = pd.DataFrame(db.target, columns = ["Test"])

#建模
lm = LinearRegression() 
lm.fit(x, y)
pred_train = lm.predict(x)
MSE_train = mse(y, pred_train)
print('-'*30)
print('MSE:',MSE_train)
print('R-square:',lm.score(x, y))

#----------------------------------------------------------------------
# 設定資料集分割為0.25，也就是拿3份資料去訓練，預測1份的資料
XTrain_1, XTest_1, yTrain_1, yTest_1 = tts(x, y, test_size = 0.25, random_state = 100)
lm = LinearRegression() 
lm.fit(XTrain_1, yTrain_1)
pred_test_1 = lm.predict(XTest_1)
MSE1 = mse(yTest_1, pred_test_1)

print('-'*30)
print('MSE1:',MSE1)
print(f'MSE1\'s R-square:{lm.score(XTest_1, yTest_1)}')

#----------------------------------------------------------------------

XTrain_2, XTest_2, yTrain_2, yTest_2 = tts(x, y, test_size = 0.2, random_state = 100)
lm = LinearRegression() 
lm.fit(XTrain_2, yTrain_2)
pred_test_2 = lm.predict(XTest_2)
MSE2 = mse(yTest_2, pred_test_2)

print('-'*30)
print('MSE2:',MSE2)
print(f'MSE2\'s R-square:{lm.score(XTest_2, yTest_2)}')

