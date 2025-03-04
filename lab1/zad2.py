import math
import random
import matplotlib.pyplot as plt

g = 9.8 # Przyspieszenie ziemskie
h = 100 # Wysokość trebusza
v0 = 50 # Początkowa prędkość pocisków

def oblicz_dystans(kąt):
    kąt_na_radiany = math.radians(kąt)
    return (v0 * math.sin(kąt_na_radiany) + math.sqrt(v0 ** 2 * math.sin(kąt_na_radiany) ** 2 + 2 * g * h)) * (v0 * math.cos(kąt_na_radiany) / g)

def rysuj_trajektorie(kąt):
    kąt_na_radiany = math.radians(kąt)

    x_values = []
    for x in range(0, int(oblicz_dystans(kąt)), 1):
        x_values.append(x)

    y_values = [(-g / (2 * v0**2 * math.cos(kąt_na_radiany)**2)) * x**2 + (math.tan(kąt_na_radiany) * x) + h for x in x_values]


    plt.plot(x_values, y_values, label=f'Trajektoria (α={kąt}°)')
    plt.xlabel('Odległość (m)')
    plt.ylabel('Wysokość (m)')
    plt.title('Trajektoria pocisku')
    plt.grid()
    plt.legend()
    plt.savefig('trajektoria.png')


cel = random.randint(50, 340)
print(f'Cel znajduje się w odległości {cel} metrów.')
print("")

proby = 0
while True:
    proby += 1
    kąt_strzalu_podany_przez_uzytkownika = float(input('Podaj kąt strzału: '))
    dystans = oblicz_dystans(kąt_strzalu_podany_przez_uzytkownika)
    print(f'Pocisk upadł na odległość {dystans:.2f} metrów.')

    if cel - 5 <= dystans <= cel + 5:
        print(f'Cel trafiony! Liczba prób: {proby}')
        rysuj_trajektorie(kąt_strzalu_podany_przez_uzytkownika)
        break
    else:
        print('Nie trafiono. Spróbuj ponownie.')











# wercja chata GPT:

# import math
# import random

# def calculate_range(v0, h, angle):
#     g = 9.81  # Przyspieszenie grawitacyjne (m/s^2)
#     alpha = math.radians(angle)  # Konwersja kąta na radiany
    
#     # Czas lotu - rozwiązanie równania ruchu pionowego
#     t1 = v0 * math.sin(alpha) / g  # Czas wznoszenia
#     t2 = math.sqrt(2 * h / g)  # Czas spadania
#     total_time = t1 + t2  # Całkowity czas lotu
    
#     # Zasięg poziomy
#     d = v0 * math.cos(alpha) * total_time
#     return d

# def main():
#     v0 = 50  # Początkowa prędkość (m/s)
#     h = 100  # Wysokość trebusza (m)
#     target = random.randint(50, 340)  # Losowa odległość celu
#     margin = 5  # Dopuszczalny margines błędu
#     attempts = 0
    
#     print(f"Cel znajduje się w odległości {target} metrów.")
    
#     while True:
#         try:
#             angle = float(input("Podaj kąt strzału (w stopniach): "))
#         except ValueError:
#             print("Proszę podać liczbę.")
#             continue
        
#         attempts += 1
#         shot_distance = calculate_range(v0, h, angle)
#         print(f"Pocisk doleciał na odległość {shot_distance:.2f} metrów.")
        
#         if target - margin <= shot_distance <= target + margin:
#             print(f"Cel trafiony! Liczba prób: {attempts}")
#             break
#         else:
#             print("Chybiony! Spróbuj ponownie.")

# if __name__ == "__main__":
#     main()



# chat zrobił to innym sposobem, uzyl innych wzorów
# Ten sposób wykonania tego zadania wydaje sie trudniejszy na pierwszy rzut oka,
# a długość kodów jest podobna