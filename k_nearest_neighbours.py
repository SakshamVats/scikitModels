import pandas as pd
import numpy as np
import seaborn as sns               # dataset from seaborn as well
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset
titanic = sns.load_dataset("titanic")
df = titanic[["pclass", "sex", "age", "fare", "survived"]].dropna()
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# split dataset into features and labels
X = df.drop("survived", axis=1)
y = df["survived"]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

# standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# test knn classifier with various values of K
'''
k_values = range(1, 21)
accuracy_scores = []

for K in k_values:
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy_scores.append(accuracy_score(y_test, y_pred))
    print(K, ":\t", accuracy_score(y_test, y_pred))
'''

# train with best value of K (8 here)
K = 8
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.2f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()