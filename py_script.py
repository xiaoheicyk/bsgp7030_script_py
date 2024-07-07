import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("-------------------------")
df = pd.read_csv("regrex1.csv")
print("Head of dataframe: \n",df.head())

print("-------------------------")
print("Shape of dataframe: \n",df.shape)

X = df[["x"]]
y = df[["y"]]
print("-------------------------")
print("Scatter plot of the original data: ")
plt.scatter(X,y,s=5,c="green")
plt.title("Scatter plot on original data")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig('py_orig.png')
plt.show()


reg = LinearRegression().fit(X, y)
print("-------------------------")
print("R squared score: %.2f",reg.score)

pred_y = reg.predict(X)

print("-------------------------")
print("Slope: %.2f",reg.coef_)


print("-------------------------")
print("Intercept: %.2f",reg.intercept_)


print("-------------------------")
print("Mean squared error: %.2f" % mean_squared_error(y,pred_y))

print("-------------------------")
print("Coefficient of determination: %.2f" %r2_score(y,pred_y))



print("-------------------------")
print("Plot of linear model:")
plt.scatter(X,y,s=5,c="green")
plt.scatter(X,pred_y,s=5,c="red")
plt.plot(X,pred_y,linestyle="dotted",c="yellow")
plt.legend(["Actual data", "Predicted data","Linear Regression"], loc="lower right")
plt.title("Scatter plot and Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig('py_lm.png')
plt.show()
