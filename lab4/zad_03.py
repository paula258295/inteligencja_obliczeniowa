import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("diabetes.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=292487)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation='relu', max_iter=500, random_state=292487)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Dokładność modelu: {accuracy:.4f}")
print("Macierz błędu:")
print(conf_matrix)



TN, FP, FN, TP = conf_matrix.ravel()

print(f"FP (False Positive): {FP}")
print(f"FN (False Negative): {FN}")

if FN > FP:
    print("Błąd False Negative (FN) jest bardziej kosztowny.")
else:
    print("Błąd False Positive (FP) jest bardziej kosztowny.")



# INNA_1
mlp_new = MLPClassifier(hidden_layer_sizes=(16, 6), activation='tanh', max_iter=500, random_state=292487)

mlp_new.fit(X_train, y_train)

y_pred_new = mlp_new.predict(X_test)
accuracy_new = accuracy_score(y_test, y_pred_new)
conf_matrix_new = confusion_matrix(y_test, y_pred_new)

print(f"Nowa dokładność modelu: {accuracy_new:.4f}")
print("Nowa macierz błędu:")
print(conf_matrix_new)




# INNA_2
mlp_new = MLPClassifier(hidden_layer_sizes=(10, 7, 3), activation='tanh', max_iter=500, random_state=292487)

mlp_new.fit(X_train, y_train)

y_pred_new = mlp_new.predict(X_test)
accuracy_new = accuracy_score(y_test, y_pred_new)
conf_matrix_new = confusion_matrix(y_test, y_pred_new)

print(f"Nowa dokładność modelu: {accuracy_new:.4f}")
print("Nowa macierz błędu:")
print(conf_matrix_new)
