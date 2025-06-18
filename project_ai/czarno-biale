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
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img[..., np.newaxis]
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

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
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

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


import matplotlib.pyplot as plt

n_images = 6
indices = np.random.choice(len(X_test), n_images, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    img = X_test[idx].reshape(IMG_SIZE, IMG_SIZE)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'{artists[y_test[idx]]}\nPred: {artists[y_pred_classes[idx]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class_names = [
    "Picasso", "van_Gogh", "Rembrandt",
    "Botticelli", "Dali", "Chagall"
]

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title("Macierz pomyłek (CNN, grayscale)")
plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes, target_names=artists))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_classes))








#                    precision    recall  f1-score   support

#     Pablo_Picasso       0.38      0.44      0.41        25
#  Vincent_van_Gogh       0.37      0.40      0.38        25
#         Rembrandt       0.93      0.52      0.67        25
# Sandro_Botticelli       0.25      0.16      0.20        25
#     Salvador_Dali       0.90      0.36      0.51        25
#      Marc_Chagall       0.33      0.72      0.46        25

#          accuracy                           0.43       150
#         macro avg       0.53      0.43      0.44       150
#      weighted avg       0.53      0.43      0.44       150

# Confusion matrix:
# [[11  5  0  2  0  7]
#  [ 3 10  1  2  1  8]
#  [ 3  4 13  3  0  2]
#  [ 8  3  0  4  0 10]
#  [ 2  4  0  1  9  9]
#  [ 2  1  0  4  0 18]]
