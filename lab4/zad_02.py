from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=292487)

print("Etykiety numeryczne:", set(y))  # 0, 1, 2
print("Odpowiadające im nazwy gatunków:", iris.target_names)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1
mlp_2_neurons = MLPClassifier(hidden_layer_sizes=(2,), max_iter=4000, random_state=292487)
mlp_2_neurons.fit(X_train, y_train)

y_pred_2 = mlp_2_neurons.predict(X_test)
accuracy_2 = accuracy_score(y_test, y_pred_2)
print("Dokładność dla sieci 4-2-1:", accuracy_2)


# 2
mlp_3_neurons = MLPClassifier(hidden_layer_sizes=(3,), max_iter=4000, random_state=292487)
mlp_3_neurons.fit(X_train, y_train)

y_pred_3 = mlp_3_neurons.predict(X_test)
accuracy_3 = accuracy_score(y_test, y_pred_3)
print("Dokładność dla sieci 4-3-1:", accuracy_3)


# 3
mlp_2_layers = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=4000, random_state=292487)
mlp_2_layers.fit(X_train, y_train)

y_pred_2_layers = mlp_2_layers.predict(X_test)
accuracy_2_layers = accuracy_score(y_test, y_pred_2_layers)
print("Dokładność dla sieci 4-3-3-1:", accuracy_2_layers)