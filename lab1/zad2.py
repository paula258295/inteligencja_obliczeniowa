import math
import random
import matplotlib.pyplot as plt

g = 9.81
h = 100
v0 = 50

def oblicz_dystans(alpha):
    alpha_rad = math.radians(alpha)
    return (v0 * math.sin(alpha_rad) + math.sqrt(v0 ** 2 * math.sin(alpha_rad) ** 2 + 2 * g * h)) * (
                v0 * math.cos(alpha_rad) / g)

def rysuj_trajektorie(alpha):
    alpha_rad = math.radians(alpha)
    vx = v0 * math.cos(alpha_rad)
    vy = v0 * math.sin(alpha_rad)

    t_max = (vy + math.sqrt(vy ** 2 + 2 * g * h)) / g
    t_values = [t * 0.01 for t in range(int(t_max / 0.01) + 1)]

    x_values = [vx * t for t in t_values]
    y_values = [vy * t - 0.5 * g * t ** 2 + h for t in t_values]

    plt.plot(x_values, y_values, label=f'Trajektoria (α={alpha}°)')
    plt.xlabel('Odległość (m)')
    plt.ylabel('Wysokość (m)')
    plt.title('Trajektoria pocisku')
    plt.grid()
    plt.legend()
    plt.savefig('trajektoria.png')
    plt.show()


cel = random.randint(50, 340)
print(f'Cel znajduje się w odległości {cel} metrów.')

proby = 0
while True:
    proby += 1
    kat = float(input('Podaj kąt strzału (w stopniach): '))
    dystans = oblicz_dystans(kat)
    print(f'Pocisk upadł na odległość {dystans:.2f} metrów.')

    if cel - 5 <= dystans <= cel + 5:
        print(f'Cel trafiony! Liczba prób: {proby}')
        rysuj_trajektorie(kat)
        break
    else:
        print('Nie trafiono. Spróbuj ponownie.')