from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# Preprocessing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with increased iterations, smaller learning rate, and a simpler architecture
neural_net = MLPRegressor(hidden_layer_sizes=(15,), activation="relu", solver="adam", learning_rate_init=0.001, max_iter=5000, random_state=69)
neural_net.fit(X_train_scaled, y_train)

y_pred = neural_net.predict(X_test_scaled)

# Test Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")
print(f"Number of iterations: {neural_net.n_iter_}")
