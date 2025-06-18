import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

artists = [
    'Pablo_Picasso', 'Vincent_van_Gogh', 'Rembrandt',
    'Sandro_Botticelli', 'Salvador_Dali', 'Marc_Chagall'
]
base_dir = 'images_balanced'
IMG_SIZE = 64

X = []
y = []
for idx, artist in enumerate(artists):
    folder = os.path.join(base_dir, artist)
    for file in os.listdir(folder):
        if file.endswith('.jpg') or file.endswith('.png'):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(idx)

X = np.array(X)
y = np.array(y)
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

num_classes = len(artists)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)


# Budowa modelu CNN do klasyfikacji obrazów malarskich
model = Sequential([
    # Pierwsza warstwa konwolucyjna: 32 filtry 3x3, aktywacja ReLU
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    # Warstwa maksymalnego pooling (zmniejsza rozmiar obrazu)
    MaxPooling2D(2, 2),
    # Druga warstwa konwolucyjna: 64 filtry 3x3, aktywacja ReLU
    Conv2D(64, (3, 3), activation='relu'),
    # Kolejna warstwa pooling
    MaxPooling2D(2, 2),
    # Spłaszczenie do wektora 1D
    Flatten(),
    # Gęsta warstwa ukryta z 128 neuronami
    Dense(128, activation='relu'),
    # Dropout (30%) – zapobiega przeuczeniu
    Dropout(0.3),
    # Warstwa wyjściowa – klasyfikacja do tylu klas, ilu artystów
    Dense(num_classes, activation='softmax')
])

# Kompilacja modelu: optymalizator Adam, funkcja straty – kategoryczna entropia krzyżowa
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Augmentacja danych treningowych – losowe przekształcenia obrazów
datagen = ImageDataGenerator(
    rotation_range=20,        # losowy obrót do 20 stopni
    width_shift_range=0.1,    # losowe przesunięcie w poziomie
    height_shift_range=0.1,   # losowe przesunięcie w pionie
    zoom_range=0.1,           # losowe powiększenie/zmniejszenie
    horizontal_flip=True      # losowe odbicie w poziomie
)
datagen.fit(X_train)

history = model.fit(
    datagen.flow(X_train, y_train_cat, batch_size=16),
    epochs=25,
    validation_data=(X_test, y_test_cat)
)

# Ewaluacja i predykcje
score = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test accuracy: {score[1]:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes, target_names=artists))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_classes))

import matplotlib.pyplot as plt

n_images = 6
indices = np.random.choice(len(X_test), n_images, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    img = X_test[idx]
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f'{artists[y_test[idx]]}\nPred: {artists[y_pred_classes[idx]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_names = [
    "Picasso", "van_Gogh", "Rembrandt",
    "Botticelli", "Dali", "Chagall"
]

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title("Macierz pomyłek (CNN)")
plt.tight_layout()
plt.show()




# aa

#                    precision    recall  f1-score   support

#     Pablo_Picasso       0.37      0.60      0.45        25
#  Vincent_van_Gogh       0.52      0.52      0.52        25
#         Rembrandt       0.89      0.68      0.77        25
# Sandro_Botticelli       0.58      0.72      0.64        25
#     Salvador_Dali       0.94      0.68      0.79        25
#      Marc_Chagall       0.62      0.40      0.49        25

#          accuracy                           0.60       150
#         macro avg       0.66      0.60      0.61       150
#      weighted avg       0.66      0.60      0.61       150

# Confusion matrix:
# [[15  5  0  4  0  1]
#  [ 6 13  1  4  0  1]
#  [ 3  0 17  4  1  0]
#  [ 4  1  1 18  0  1]
#  [ 2  2  0  1 17  3]
#  [11  4  0  0  0 10]]
