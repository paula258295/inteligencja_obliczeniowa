import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier

df = pd.read_csv("iris.csv")

train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=292487)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

for k in [3, 5, 11]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_inputs, train_classes)
    knn_predictions = knn.predict(test_inputs)
    knn_accuracy = accuracy_score(test_classes, knn_predictions)
    print(f"Dokładność klasyfikatora {k}-NN: {knn_accuracy * 100:.2f}%")
    
    knn_conf_matrix = confusion_matrix(test_classes, knn_predictions)
    sns.heatmap(knn_conf_matrix, annot=True, cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_)
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista klasa")
    plt.title(f"Macierz błędów - {k}-NN")
    plt.savefig(f"zad3_confusion_matrix_{k}NN.png")
    plt.clf()


nb = GaussianNB()
nb.fit(train_inputs, train_classes)
nb_predictions = nb.predict(test_inputs)
nb_accuracy = accuracy_score(test_classes, nb_predictions)
print(f"Dokładność klasyfikatora Naive Bayes: {nb_accuracy * 100:.2f}%")

nb_conf_matrix = confusion_matrix(test_classes, nb_predictions)
sns.heatmap(nb_conf_matrix, annot=True, cmap="Blues", xticklabels=nb.classes_, yticklabels=nb.classes_)
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywista klasa")
plt.title("Macierz błędów - Naive Bayes")
plt.savefig("zad3_confusion_matrix_nb.png")
plt.clf()


dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(train_inputs, train_classes)
y_pred_dummy = dummy.predict(test_inputs)
accuracy_dummy = accuracy_score(test_classes, y_pred_dummy)
print(f'Dummy Classifier - Accuracy: {accuracy_dummy * 100:.2f}%\n')