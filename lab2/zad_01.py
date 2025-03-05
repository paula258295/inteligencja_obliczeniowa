# import pandas as pd

# df = pd.read_csv("iris1.csv") 
# print(df)

# print(df.values) 

# #wszystkie wiersze, kolumna nr 0 
# print(df.values[:, 0]) 
# #wiersze od 5 do 10, wszystkie kolumny 
# print(df.values[5:11, :]) 
# #dane w komórce [1,4] 
# print(df.values[1, 4]) 

import pandas as pd
import numpy as np

file_path = "iris_with_errors.csv"  # Podmień na rzeczywistą ścieżkę pliku
df = pd.read_csv(file_path)

missing_values = df.isnull().sum()
print("Brakujące wartości w każdej kolumnie:")
print(missing_values)

# numeric_cols = df.select_dtypes(include=[np.number]).columns # NA lub -
# for col in numeric_cols:
#     df[col] = df[col].apply(lambda x: np.nan if x <= 0 or x > 15 else x)
# df.fillna({col: df[col].median() for col in numeric_cols}, inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df.loc[(df[col] <= 0) | (df[col] > 15), col] = np.nan  # Zamiana błędnych wartości na NaN
df.fillna(df[numeric_cols].median(), inplace=True)

# valid_species = {"Setosa", "Versicolor", "Virginica"}
# df["variety"] = df["variety"].str.strip()
# invalid_species = df[~df["variety"].isin(valid_species)]["variety"].unique()
# print("Niepoprawne nazwy gatunków:", invalid_species)
fixed_species = {"setsoa": "Setosa", "versicolour": "Versicolor", "virgnica": "Virginica"}
for wrong, correct in fixed_species.items():
    df.loc[df["variety"].str.lower() == wrong, "variety"] = correct

fixed_species = {"Setsoa": "setosa", "Versicolr": "Versicolour", "virgnica": "virginica"}
df["variety"] = df["variety"].replace(fixed_species)

df.to_csv("iris_cleaned.csv", index=False)
print("Dane zostały oczyszczone i zapisane jako 'iris_cleaned.csv'.")