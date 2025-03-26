import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

df = pd.read_csv("iris.csv")

train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=292487)

print('--- treningowy ---')
print(train_set)

print('--- testowy ---')
print(test_set)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

clf = DecisionTreeClassifier()
clf.fit(train_inputs, train_classes)

from sklearn.tree import export_text
print(export_text(clf, feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"]))

plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"], class_names=clf.classes_, filled=True)
plt.savefig("zad2_decision_tree.png")

predictions = clf.predict(test_inputs)
accuracy = accuracy_score(test_classes, predictions)
print(f"Dokładność klasyfikatora DD: {accuracy * 100:.2f}%")
