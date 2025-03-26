import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris.csv")

train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=292487)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]


def classify_iris(sl, sw, pl, pw): 
    if sl > 4: 
        return("Setosa") 
    elif pl <= 5: 
        return("Virginica") 
    else: 
        return("Versicolor") 
# Poprawnie sklasyfikowane: 13 na 45
# Dokładność klasyfikatora: 28.89%




# ZBIÓR TRENINGOWY 
train_set_sorted = train_set[train_set[:, 4].argsort()]

print("Posortowany zbiór testowy: ", train_set_sorted)

# def classify_iris(sl, sw, pl, pw):
#     # Setosa: krótki sepal_length (sl) i mały petal_length (pl)
#     if pl < 2:
#         return "Setosa"
    
#     # Virginica: duży petal_length (pl) i petal_width (pw)
#     elif pl >= 5 and pw >= 1.8:
#         return "Virginica"
    
#     # Pozostałe przypadki: Versicolor
#     else:
#         return "Versicolor"
# # Poprawnie sklasyfikowane: 43 na 45
# # Dokładność klasyfikatora: 95.56%



good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    sl, sw, pl, pw = test_set[i, 0], test_set[i, 1], test_set[i, 2], test_set[i, 3]
    true_class = test_set[i, 4]
    
    if classify_iris(sl, sw, pl, pw) == true_class:
        good_predictions += 1

total_accuracy = good_predictions / len * 100
print(f"Poprawnie sklasyfikowane: {good_predictions} na {len}")
print(f"Dokładność klasyfikatora: {total_accuracy:.2f}%")

# print(train_set) 
# print('---------------------------')
# print(test_set)