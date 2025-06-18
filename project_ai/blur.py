import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator



output_dir = 'images_blur'
IMG_SIZE = 64

artists = [
    'Pablo_Picasso', 'Vincent_van_Gogh', 'Rembrandt',
    'Sandro_Botticelli', 'Salvador_Dali', 'Marc_Chagall'
]

X = []
y = []
for idx, artist in enumerate(artists):
    folder = os.path.join(output_dir, artist)
    print(f"Wczytuję z folderu: {folder}")
    for file in os.listdir(folder):
        if file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Nie można wczytać: {img_path}")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"Nie można wczytać: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(idx)
print("Liczba obrazów:", len(X))

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

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
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







#                    precision    recall  f1-score   support

#     Pablo_Picasso       0.42      0.40      0.41        25
#  Vincent_van_Gogh       0.32      0.32      0.32        25
#         Rembrandt       0.75      0.72      0.73        25
# Sandro_Botticelli       0.78      0.56      0.65        25
#     Salvador_Dali       0.76      0.52      0.62        25
#      Marc_Chagall       0.43      0.72      0.54        25

#          accuracy                           0.54       150
#         macro avg       0.58      0.54      0.55       150
#      weighted avg       0.58      0.54      0.55       150

# Confusion matrix:
# [[10  5  2  1  1  6]
#  [ 6  8  3  1  1  6]
#  [ 3  1 18  2  1  0]
#  [ 1  5  1 14  1  3]
#  [ 0  3  0  0 13  9]
#  [ 4  3  0  0  0 18]]
