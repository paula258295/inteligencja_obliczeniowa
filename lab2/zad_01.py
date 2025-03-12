import pandas as pd
import numpy as np

file_path = "iris_with_errors.csv"
data = pd.read_csv(file_path)

data.replace(["-", "NA"], np.nan, inplace=True)

data.iloc[:, :-1] = data.iloc[:, :-1].apply(pd.to_numeric)

missing_values = data.isnull().sum()
print("Brakujące wartości NA lub - w kolumnach:")
print(missing_values)

# b)
for col in data.columns[:-1]:
    valid_values = data[col][(data[col] > 0) & (data[col] < 15)]
    median_value = valid_values.median()
    data[col] = np.where((data[col] <= 0) | (data[col] >= 15) | (data[col].isnull()), median_value, data[col])

# c)
valid_species = {"Setosa", "Versicolor", "Virginica"}

invalid_species = data[~data["variety"].isin(valid_species)]["variety"].unique()

if len(invalid_species) > 0:
    print("Niepoprawne nazwy gatunków w bazie danych:")
    print(invalid_species)
else:
    print("Wszystkie nazwy gatunków są poprawne.")

species_correction = {
    "setosa": "Setosa",
    "Versicolour": "Versicolor",
    "virginica": "Virginica"
}

data["variety"] = data["variety"].replace(species_correction)

if data["variety"].isin(valid_species).all():
    print("Wszystkie nazwy gatunków zostały poprawione.")
else:
    print("Pozostały jeszcze błędne wpisy.")

output_file = "iris_cleaned.csv"
data.to_csv(output_file, index=False)
print(f"Poprawione dane zapisano do pliku: {output_file}")

print("\nPoprawione dane:")
print(data)