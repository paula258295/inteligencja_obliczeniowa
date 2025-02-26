import datetime
import math

print(datetime.date.today())

name = input("Your name: ")
year_of_birth = int(input("Your year of birth: "))
month_of_birth = int(input("Your month of birth: "))
day_of_birth = int(input("Your day of birth: "))

birth = datetime.date(year_of_birth, month_of_birth, day_of_birth)
days = (datetime.date.today() - birth).days

print("Hej " + name + ", żyjesz już " + str(days) + " dni!")

def oblicz_biorytm(dzien_zycia, okres):
    return math.sin((2 * math.pi * dzien_zycia) / okres)

fizyczna_fala = oblicz_biorytm(days, 23)
emocjonalna_fala = oblicz_biorytm(days, 28)
intelektualna_fala = oblicz_biorytm(days, 33)

print("Wyniki na dzień dzisiejszy: ")
print(f"Fizyczna fala: {fizyczna_fala:.2f}, Emocjonalna fala: {emocjonalna_fala:.2f}, Intelektualna fala: {intelektualna_fala:.2f}")
print(" ")

fizyczna_fala_jutro = oblicz_biorytm(days + 1, 23)
emocjonalna_fala_jutro = oblicz_biorytm(days + 1, 28)
intelektualna_fala_jutro = oblicz_biorytm(days + 1, 33)

if fizyczna_fala >= 0.5:
    print("Świetny wynik fizycznej fali!")

if emocjonalna_fala >= 0.5:
    print("Świetny wynik emocjonalnej fali!")

if intelektualna_fala >= 0.5:
    print("Świetny wynik intelektualnej fali!")

if (intelektualna_fala <= -0.5 or emocjonalna_fala <= -0.5 or fizyczna_fala <= -0.5):
    if (intelektualna_fala_jutro > 0.5 or emocjonalna_fala_jutro > 0.5 or fizyczna_fala_jutro > 0.5):
        print("Nie wszystkie wyniki wyszły pozytywne, jutro bedzie lepiej!")


print("")
print(f"Wyniki jutro: {fizyczna_fala_jutro:.2f}, Emocjonalna fala: {emocjonalna_fala_jutro:.2f}, Intelektualna fala: {intelektualna_fala_jutro:.2f}")


# zajelo mi okolo 40min





# odp chata (5sec):
# import math
# import datetime
#
#
# def oblicz_biorytm(dzien_zycia, okres):
#     return math.sin((2 * math.pi * dzien_zycia) / okres)
#
#
# def main():
#     # Pobranie danych od użytkownika
#     imie = input("Podaj swoje imię: ")
#     rok = int(input("Podaj rok urodzenia: "))
#     miesiac = int(input("Podaj miesiąc urodzenia: "))
#     dzien = int(input("Podaj dzień urodzenia: "))
#
#     # Obliczenie liczby dni życia
#     data_urodzenia = datetime.date(rok, miesiac, dzien)
#     dzisiaj = datetime.date.today()
#     dzien_zycia = (dzisiaj - data_urodzenia).days
#
#     # Obliczenie biorytmów
#     fizyczny = oblicz_biorytm(dzien_zycia, 23)
#     emocjonalny = oblicz_biorytm(dzien_zycia, 28)
#     intelektualny = oblicz_biorytm(dzien_zycia, 33)
#
#     # Obliczenie biorytmów na następny dzień
#     fizyczny_jutro = oblicz_biorytm(dzien_zycia + 1, 23)
#     emocjonalny_jutro = oblicz_biorytm(dzien_zycia + 1, 28)
#     intelektualny_jutro = oblicz_biorytm(dzien_zycia + 1, 33)
#
#     # Powitanie i wyniki
#     print(f"\nWitaj, {imie}! Dziś jest {dzien_zycia} dzień Twojego życia.")
#     print(f"Twoje dzisiejsze biorytmy:")
#     print(f"- Fizyczny: {fizyczny:.2f}")
#     print(f"- Emocjonalny: {emocjonalny:.2f}")
#     print(f"- Intelektualny: {intelektualny:.2f}")
#
#     # Interpretacja wyników
#     def interpretuj_biorytm(nazwa, wartosc, wartosc_jutro):
#         if wartosc > 0.5:
#             print(f"Twój {nazwa} poziom jest wysoki! Świetnie się dziś czujesz!")
#         elif wartosc < -0.5:
#             print(f"Twój {nazwa} poziom jest niski. To może być ciężki dzień.")
#             if wartosc_jutro > wartosc:
#                 print("Nie martw się. Jutro będzie lepiej!")
#             else:
#                 print("Niestety, jutro może nie być dużo lepiej. Odpocznij i dbaj o siebie!")
#
#     interpretuj_biorytm("fizyczny", fizyczny, fizyczny_jutro)
#     interpretuj_biorytm("emocjonalny", emocjonalny, emocjonalny_jutro)
#     interpretuj_biorytm("intelektualny", intelektualny, intelektualny_jutro)
#
#
# if __name__ == "__main__":
#     main()

