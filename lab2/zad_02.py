import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("iris1.csv")

df.columns = df.columns.str.strip()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2) # dane do 2 głównych składowych (PC1, PC2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["variety"] = y

df_pca.to_csv("iris_pca.csv", index=False)
print("Dane skompresowane do 2 wymiarów i zapisane do 'iris_pca.csv'.")


# Wykres
plt.figure(figsize=(8, 6))
colors = {'Setosa': 'blue', 'Versicolor': 'orange', 'Virginica': 'green'}
for variety in df_pca["variety"].unique():
    subset = df_pca[df_pca["variety"] == variety]
    plt.scatter(subset["PC1"], subset["PC2"], label=variety, color=colors[variety], alpha=0.7)

plt.xlabel("Główna składowa 1 (PC1)")
plt.ylabel("Główna składowa 2 (PC2)")
plt.title("Wizualizacja PCA (redukcja do 2 wymiarów)")
plt.legend()
plt.grid()
plt.savefig("pca_wizualizacja.png")

# Każda odmiana tworzy osobne skupisko - PCA dobrze oddziela klasy, można je łatwo klasyfikować
# Analiza pozwoliła zobaczyć, czy dane po PCA wciąż dobrze odzwierciedlają różnice między odmianami irysów.



explained_variance_ratio = pca.explained_variance_ratio_
print("Wariancja dla PC1 i PC2:", explained_variance_ratio)
print("Łączna wariancja:", explained_variance_ratio.sum())

pca_full = PCA().fit(X_scaled)
cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
num_components = (cumulative_variance >= 0.95).argmax() + 1
print(f"Liczba wymaganych kolumn do zachowania 95% wariancji: {num_components}")
print(f"Można usunąć {X.shape[1] - num_components} kolumny i nadal zachować 95% informacji.")

# Dwie pierwsze główne składowe (PC1 i PC2) zachowują 95,81% całkowitej wariancji. 
# Oznacza to, że dwie pozostałe kolumny oryginalnych danych można usunąć, 
# ponieważ ich wkład w informację jest mniejszy niż 5%.