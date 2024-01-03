import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt

db = datasets.load_diabetes()

#print(db.feature_names)

x_1 = pd.DataFrame(db.data, columns = db.feature_names)
x_2 = x_1.iloc[:,0:4]
y = pd.DataFrame(db.target, columns = ["Predicted Quantitatjve Measure"])


#建模
lm_1 = LinearRegression() 
lm_1.fit(x_1, y)
lm_2 = LinearRegression() 
lm_2.fit(x_2,y)

#畫圖
pre_tar1 = lm_1.predict(x_1) 
plt.scatter(y, pre_tar1)
plt.xlabel("Quantitative Measure")
plt.ylabel("Predicted Quantitive Measure")
plt.title("Quantitative Measure vs Predicted Quantitative Measure")
plt.show()


pre_tar2 = lm_2.predict(x_2)
plt.scatter(y, pre_tar2)
plt.xlabel("Quantitative Measure")
plt.ylabel("Predicted Quantitive Measure")
plt.title("Quantitative Measure vs Predicted Quantitative Measure")
plt.show()