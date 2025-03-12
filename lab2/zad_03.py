import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("iris1.csv")

# rozkład przed skalowaniem
stats = df[['sepal.length', 'sepal.width']].describe().T[['min', 'max', 'mean', 'std']]
print("Podstawowe statystyki:\n", stats)

# Skalowanie danych
scaler_minmax = MinMaxScaler()
df[['sepal.length_minmax', 'sepal.width_minmax']] = scaler_minmax.fit_transform(df[['sepal.length', 'sepal.width']])

scaler_zscore = StandardScaler()
df[['sepal.length_zscore', 'sepal.width_zscore']] = scaler_zscore.fit_transform(df[['sepal.length', 'sepal.width']])


colors = {'Setosa': 'blue', 'Versicolor': 'orange', 'Virginica': 'green'}
df['color'] = df['variety'].map(colors)

# Wykresy
plt.figure(figsize=(18, 5))

# Wykres 1: Oryginalne dane
plt.subplot(1, 3, 1)
plt.scatter(df['sepal.length'], df['sepal.width'], c=df['color'])
plt.title('Original Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')

# Wykres 2: Zeskalowane Z-Score
plt.subplot(1, 3, 2)
plt.scatter(df['sepal.length_zscore'], df['sepal.width_zscore'], c=df['color'])
plt.title('Z-Score Scaled Dataset')
plt.xlabel('Sepal Length (Z-Score)')
plt.ylabel('Sepal Width (Z-Score)')

# Wykres 3: Znormalizowane Min-Max
plt.subplot(1, 3, 3)
plt.scatter(df['sepal.length_minmax'], df['sepal.width_minmax'], c=df['color'])
plt.title('Min-Max Normalized Dataset')
plt.xlabel('Sepal Length (Min-Max)')
plt.ylabel('Sepal Width (Min-Max)')

plt.tight_layout()
plt.savefig("zad3.png")


# Co możemy powiedzieć o danych?
# Oryginalne dane: mają naturalny rozkład wartości (bez zmian).
# Normalizacja Min-Max: wszystkie wartości są w zakresie od 0 do 1.
# Standaryzacja Z-Score: wartości mają średnią 0 i odchylenie standardowe 1.