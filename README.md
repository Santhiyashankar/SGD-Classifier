# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and create a DataFrame with feature names and target labels.
2. Separate the data into features (`X`) and target (`y`).
3. Split the data into training and testing sets with an 80-20 ratio.
4. Initialize a Stochastic Gradient Descent (SGD) classifier and train it on the training data.
5. Predict the target values for the test set.
6. Calculate and display the model's accuracy.
7. Compute and display the confusion matrix for the predictions.

## Program:
```Python
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SANTHIYA S
RegisterNumber: 212223220098
*/

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

```

## Output:
![image](https://github.com/user-attachments/assets/e5d375e1-3d23-4102-8e5c-40139df6b2d2)

![image](https://github.com/user-attachments/assets/8fd94b4e-20db-4489-a082-8f70b3007070)

![image](https://github.com/user-attachments/assets/7478569b-1f31-44d1-b8bd-a36762b97297)

![image](https://github.com/user-attachments/assets/442bd768-c880-4d8e-ad07-cc6f3faed611)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
