import numpy as np                                          # for handling arrays
import pandas as pd                                         # to manipulate structured data
import matplotlib.pyplot as plt                             # to visualize data

from sklearn.datasets import fetch_california_housing                    # load Boston Housing dataset     
from sklearn.model_selection import train_test_split        # to split into training and test data
from sklearn.linear_model import LinearRegression           # machine learning model used
from sklearn.metrics import mean_squared_error              # to evaluate model performance

# load dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target # add target variable

# display first few rows
print(df.head())

# prepare feature and label
X = df[["AveRooms"]]
y = df["PRICE"]

# split data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# predict house prices based on rooms
y_pred = model.predict(X_test)

# evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# visualize results using mpl
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Line")
plt.xlim(0, 10)
plt.xlabel("Average number of rooms (AveRooms)")
plt.ylabel("House price ($100,000s)")
plt.legend()
plt.title("Linear Regression: Rooms vs Price")
plt.show()