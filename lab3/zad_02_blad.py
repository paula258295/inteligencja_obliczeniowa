import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("iris.csv")

train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=292487)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = DecisionTreeClassifier()
clf.fit(train_inputs, train_classes)

predictions = clf.predict(test_inputs)
accuracy = accuracy_score(test_classes, predictions)

conf_matrix = confusion_matrix(test_classes, predictions)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywista klasa")
plt.title("Macierz błędów - DD")
plt.savefig("zad2_confusion_matrix_dd.png")