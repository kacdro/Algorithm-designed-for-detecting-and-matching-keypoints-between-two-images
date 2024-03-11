import argparse
import colorsys
from typing import List
import cv2
from PIL import Image as PILImage
import copy

import math
import numpy as np

MAKS_ITERACJI_UDOSKONALANIA = 5
SIGMA_MIN = 0.8
MIN_PIKS_DYSTANS = 0.5
SIGMA_IN = 0.5
L_OKT = 8
S_OKT = 3
KONTRAST_DOG = 0.015
PROG_DLA_KRAWEDZI = 10

LICZBA_KOSZYKOW = 36
LAMBDA_ORIENTACJA = 1.5
LICZBA_HIST = 4
LICZBA_ORIENTACJI = 8
LAMBDA_DEKSRYPTOR = 6

PROG_ABSOLUTNY = 350
PROG_RELATYWNY = 0.7

from PIL import Image as PILImage
import numpy as np
from enum import Enum


class Interpolacja(Enum):
    DWULINIOWA = 1
    NAJBLIZSZEGO_SASIADA = 2


class Obraz:
    def __init__(self, sciezka_do_pliku= None, szer=0, wys=0, kanal=0):
        if sciezka_do_pliku:
            self.wczytaj_obraz(sciezka_do_pliku)
        else:
            self.szerokosc = szer
            self.wysokosc = wys
            self.liczba_kanalow = kanal
            self.rozmiar = szer * wys * kanal
            self.piksele = np.zeros((wys, szer, kanal), dtype=np.float32) if self.rozmiar > 0 else None

    def wczytaj_obraz(self, sciezka_do_obrazu):
        obraz = PILImage.open(sciezka_do_obrazu)
        dane_obrazu = np.asarray(obraz, dtype=np.float32) / 255.0
        if len(dane_obrazu.shape) == 2:
            dane_obrazu = np.expand_dims(dane_obrazu, axis=-1)
        self.piksele = dane_obrazu
        self.wysokosc, self.szerokosc = dane_obrazu.shape[:2]
        self.liczba_kanalow = dane_obrazu.shape[2] if len(dane_obrazu.shape) > 2 else 1
        self.rozmiar = self.szerokosc * self.wysokosc * self.liczba_kanalow

    def wyswietl_informacje_o_obrazie(self):
        print(
            f"Szerokosc obrazu: {self.szerokosc} px, Wysokosc: {self.wysokosc} px, Liczba kanalow: {self.liczba_kanalow}"
        )

    def kopiuj_obraz(self):
        return copy.copy(self)

    def zapisz_do_pliku(self, sciezka_z_nazwa_pliku):

        if self.piksele.dtype != np.uint8:
            dane_do_pliku = (self.piksele * 255).clip(0, 255).astype(np.uint8)
        else:
            dane_do_pliku = self.piksele

        try:

            if self.liczba_kanalow == 1:
                dane_do_pliku = dane_do_pliku.reshape((self.wysokosc, self.szerokosc))
            else:
                dane_do_pliku = dane_do_pliku.reshape((self.wysokosc, self.szerokosc, self.liczba_kanalow))

            obraz = PILImage.fromarray(dane_do_pliku)
            obraz.save(sciezka_z_nazwa_pliku)
            return True

        except Exception as e:
            print(f"Nie mozna zapisac obrazu: {sciezka_z_nazwa_pliku}\nError: {e}")
            return False

    def ustaw_pixel(self, x, y, k, wartosc):
        self.piksele[y, x, k] = wartosc

    def pobierz_pixel(self, x, y, k):

        x = max(0, min(x, self.szerokosc - 1))
        y = max(0, min(y, self.wysokosc - 1))

        return self.piksele[y, x, k]

    def mapuj_koordynaty_pixeli(self, nowy_max, obecny_max, parametr):
        a = nowy_max / obecny_max
        b = -0.5 + a * 0.5
        return a * parametr + b

    def interpolacja_nn(self, obraz, x, y, k):
        return obraz.pobierz_pixel(round(x), round(y), k)

    def zmien_wielkosc_obrazu(self, nowa_szer, nowa_wys, metoda):

        obraz_z_nowym_rozmiarem = Obraz(szer=nowa_szer, wys=nowa_wys, kanal=self.liczba_kanalow)
        wartosc_px = 0
        for x in range(nowa_szer):
            for y in range(nowa_wys):
                for k in range(self.liczba_kanalow):

                    stare_x = self.mapuj_koordynaty_pixeli(self.szerokosc, nowa_szer, x)
                    stare_y = self.mapuj_koordynaty_pixeli(self.wysokosc, nowa_wys, y)

                    if metoda == Interpolacja.DWULINIOWA:
                        wartosc_px = self.interpolacja_dwuliniowa(self, stare_x, stare_y, k)
                    elif metoda == Interpolacja.NAJBLIZSZEGO_SASIADA:
                        wartosc_px = self.interpolacja_nn(self, stare_x, stare_y, k)

                    obraz_z_nowym_rozmiarem.ustaw_pixel(x, y, k, wartosc_px)

        return obraz_z_nowym_rozmiarem

    def interpolacja_dwuliniowa(self, obraz, x, y, k):

        x_podloga = math.floor(x)
        y_podloga = math.floor(y)
        x_sufit = x_podloga + 1
        y_sufit = y_podloga + 1

        p1 = obraz.pobierz_pixel(x_podloga, y_podloga, k)
        p2 = obraz.pobierz_pixel(x_sufit, y_podloga, k)
        p3 = obraz.pobierz_pixel(x_podloga, y_sufit, k)
        p4 = obraz.pobierz_pixel(x_sufit, y_sufit, k)

        q1 = (y_sufit - y) * p1 + (y - y_podloga) * p3
        q2 = (y_sufit - y) * p2 + (y - y_podloga) * p4

        return (x_sufit - x) * q1 + (x - x_podloga) * q2


class PunktKluczowy:
    def __init__(
            self,
            i: int,
            j: int,
            oktawy: int,
            skala: int,
            x: float,
            y: float,
            sigma: float,
            wartosc_ekstremum: float,
    ):
        self.i = i
        self.j = j
        self.oktawy = oktawy
        self.skala = skala
        self.x = x
        self.y = y
        self.sigma = sigma
        self.wartosc_ekstremum = wartosc_ekstremum
        self.deskryptor = [0] * 128


def obraz_rgb_do_szarosci(obraz):
    obraz_w_skali_szarosci = Obraz(szer=obraz.szerokosc, wys=obraz.wysokosc, kanal=1)

    for y in range(obraz.wysokosc):
        for x in range(obraz.szerokosc):
            pixel_czerwony = obraz.pobierz_pixel(x, y, 0)
            pixel_zielony = obraz.pobierz_pixel(x, y, 1)
            pixel_niebieski = obraz.pobierz_pixel(x, y, 2)

            odcien_szarosci = 0.299 * pixel_czerwony + 0.587 * pixel_zielony + 0.114 * pixel_niebieski

            obraz_w_skali_szarosci.ustaw_pixel(x, y, 0, odcien_szarosci)

    return obraz_w_skali_szarosci


class PiramidaSkal:
    def __init__(self, liczba_oktaw, obrazy_na_oktawe, oktawy=None):
        self.liczba_oktaw = liczba_oktaw
        self.obrazy_na_oktawe = obrazy_na_oktawe
        self.oktawy = (
            oktawy if oktawy is not None else [[] for _ in range(liczba_oktaw)]
        )


def rozmycie_gaussa(obraz, sigma):
    assert obraz.liczba_kanalow == 1, "Obraz musi miec tylko jeden kanal"

    rozmiar = int(math.ceil(6 * sigma))
    if rozmiar % 2 == 0:
        rozmiar += 1
    srodek = rozmiar // 2

    jadro = Obraz(szer=rozmiar, wys=1, kanal=1)
    suma = 0.0
    for k in range(-rozmiar // 2, rozmiar // 2 + 1):
        wartosc = math.exp(-(k * k) / (2 * sigma * sigma))
        jadro.ustaw_pixel(srodek + k, 0, 0, wartosc)
        suma += wartosc
    for k in range(rozmiar):
        obecna_wartosc = jadro.pobierz_pixel(k, 0, 0)
        jadro.ustaw_pixel(k, 0, 0, obecna_wartosc / suma)

    tymczasowy_obraz = Obraz(szer=obraz.szerokosc, wys=obraz.wysokosc, kanal=1)
    przefiltrowany_obraz = Obraz(szer=obraz.szerokosc, wys=obraz.wysokosc, kanal=1)

    for x in range(obraz.szerokosc):
        for y in range(obraz.wysokosc):
            suma = 0.0
            for k in range(rozmiar):
                dy = -srodek + k

                suma += obraz.pobierz_pixel(x, y + dy, 0) * jadro.pobierz_pixel(k, 0, 0)
            tymczasowy_obraz.ustaw_pixel(x, y, 0, suma)

    for x in range(obraz.szerokosc):
        for y in range(obraz.wysokosc):
            suma = 0.0
            for k in range(rozmiar):
                dx = -srodek + k

                suma += tymczasowy_obraz.pobierz_pixel(x + dx, y, 0) * jadro.pobierz_pixel(k, 0, 0)
            przefiltrowany_obraz.ustaw_pixel(x, y, 0, suma)

    return przefiltrowany_obraz


def tworzenie_piramidy(
        obraz: Obraz, sigma_min=SIGMA_MIN, liczba_oktaw=L_OKT, skala_na_oktawe=S_OKT
):
    sigma_bazowa = sigma_min / MIN_PIKS_DYSTANS
    obraz_bazowy = obraz.zmien_wielkosc_obrazu(obraz.szerokosc * 2, obraz.wysokosc * 2, Interpolacja.DWULINIOWA)

    roznica_sigma = math.sqrt(sigma_bazowa ** 2 - 1.0)
    obraz_bazowy = rozmycie_gaussa(obraz_bazowy, roznica_sigma)

    obrazy_na_oktawe = skala_na_oktawe + 3

    k = 2 ** (1.0 / skala_na_oktawe)
    wartosc_sigma = [sigma_bazowa]
    for i in range(1, obrazy_na_oktawe):
        sigma_poprzednia = sigma_bazowa * k ** (i - 1)
        wartosc_calkowita_sigma = k * sigma_poprzednia
        wartosc_sigma.append(math.sqrt(wartosc_calkowita_sigma ** 2 - sigma_poprzednia ** 2))

    piramidaSkal = PiramidaSkal(liczba_oktaw, obrazy_na_oktawe)
    for i in range(liczba_oktaw):
        piramidaSkal.oktawy[i].append(obraz_bazowy)
        for j in range(1, len(wartosc_sigma)):
            poprzedni_obraz = piramidaSkal.oktawy[i][-1]
            piramidaSkal.oktawy[i].append(rozmycie_gaussa(poprzedni_obraz, wartosc_sigma[j]))

        nastepny_bazowy_obraz = piramidaSkal.oktawy[i][obrazy_na_oktawe - 3]
        obraz_bazowy = nastepny_bazowy_obraz.zmien_wielkosc_obrazu(
            nastepny_bazowy_obraz.szerokosc // 2, nastepny_bazowy_obraz.wysokosc // 2, Interpolacja.NAJBLIZSZEGO_SASIADA
        )

    return piramidaSkal


def tworzenie_piramidy_dog(obraz_piramidy):
    piramida_dog = PiramidaSkal(
        liczba_oktaw=obraz_piramidy.liczba_oktaw,
        obrazy_na_oktawe=obraz_piramidy.obrazy_na_oktawe - 1,
    )

    for i in range(piramida_dog.liczba_oktaw):
        for j in range(1, obraz_piramidy.obrazy_na_oktawe):

            obraz_1 = obraz_piramidy.oktawy[i][j]
            obraz_2 = obraz_piramidy.oktawy[i][j - 1]

            czego_brakuje = []
            if obraz_1 is None:
                czego_brakuje.append("img1")
            if obraz_2 is None:
                czego_brakuje.append("img2")
            if obraz_1 and obraz_1.piksele is None:
                czego_brakuje.append("img1.piksele")
            if obraz_2 and obraz_2.piksele is None:
                czego_brakuje.append("img2.piksele")

            if czego_brakuje:
                continue

            roznica = Obraz(szer=obraz_1.szerokosc, wys=obraz_1.wysokosc, kanal=obraz_1.liczba_kanalow)
            roznica.piksele = obraz_1.piksele - obraz_2.piksele
            piramida_dog.oktawy[i].append(roznica)

    return piramida_dog


def punkt_jest_ekstremum(oktawa, skala, x, y):
    if 0 < skala < len(oktawa) - 1:

        obraz = oktawa[skala]
        poprzedni = oktawa[skala - 1]
        nastepny = oktawa[skala + 1]

        czy_jest_min = True
        czy_jest_max = True
        wartosc = obraz.pobierz_pixel(x, y, 0)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:

                if dx == 0 and dy == 0:
                    continue

                sasiad = poprzedni.pobierz_pixel(x + dx, y + dy, 0)
                if sasiad >= wartosc:
                    czy_jest_max = False
                if sasiad <= wartosc:
                    czy_jest_min = False

                sasiad = nastepny.pobierz_pixel(x + dx, y + dy, 0)
                if sasiad >= wartosc:
                    czy_jest_max = False
                if sasiad <= wartosc:
                    czy_jest_min = False

                sasiad = obraz.pobierz_pixel(x + dx, y + dy, 0)
                if sasiad >= wartosc:
                    czy_jest_max = False
                if sasiad <= wartosc:
                    czy_jest_min = False

                if not czy_jest_min and not czy_jest_max:
                    return False

        return czy_jest_min or czy_jest_max
    else:
        return False


def aproksymacja_kwadratowa(punkty_kluczowe, oktawa, skala):
    obraz = oktawa[skala]
    poprzedni = oktawa[skala - 1]
    nastepny = oktawa[skala + 1]

    x, y = punkty_kluczowe.i, punkty_kluczowe.j

    g1 = (nastepny.pobierz_pixel(x, y, 0) - poprzedni.pobierz_pixel(x, y, 0)) * 0.5
    g2 = (obraz.pobierz_pixel(x + 1, y, 0) - obraz.pobierz_pixel(x - 1, y, 0)) * 0.5
    g3 = (obraz.pobierz_pixel(x, y + 1, 0) - obraz.pobierz_pixel(x, y - 1, 0)) * 0.5

    h11 = nastepny.pobierz_pixel(x, y, 0) + poprzedni.pobierz_pixel(x, y, 0) - 2 * obraz.pobierz_pixel(x, y, 0)
    h22 = (
            obraz.pobierz_pixel(x + 1, y, 0)
            + obraz.pobierz_pixel(x - 1, y, 0)
            - 2 * obraz.pobierz_pixel(x, y, 0)
    )
    h33 = (
            obraz.pobierz_pixel(x, y + 1, 0)
            + obraz.pobierz_pixel(x, y - 1, 0)
            - 2 * obraz.pobierz_pixel(x, y, 0)
    )
    h12 = (
                  nastepny.pobierz_pixel(x + 1, y, 0)
                  - nastepny.pobierz_pixel(x - 1, y, 0)
                  - poprzedni.pobierz_pixel(x + 1, y, 0)
                  + poprzedni.pobierz_pixel(x - 1, y, 0)
          ) * 0.25
    h13 = (
                  nastepny.pobierz_pixel(x, y + 1, 0)
                  - nastepny.pobierz_pixel(x, y - 1, 0)
                  - poprzedni.pobierz_pixel(x, y + 1, 0)
                  + poprzedni.pobierz_pixel(x, y - 1, 0)
          ) * 0.25
    h23 = (
                  obraz.pobierz_pixel(x + 1, y + 1, 0)
                  - obraz.pobierz_pixel(x + 1, y - 1, 0)
                  - obraz.pobierz_pixel(x - 1, y + 1, 0)
                  + obraz.pobierz_pixel(x - 1, y - 1, 0)
          ) * 0.25

    macierz_hessego = np.array([[h11, h12, h13], [h12, h22, h23], [h13, h23, h33]])

    blad = 1e-2
    H_zregularyzowany = macierz_hessego + blad * np.eye(macierz_hessego.shape[0])

    g = np.array([g1, g2, g3])

    try:
        H_odwrotna = np.linalg.inv(H_zregularyzowany)
    except np.linalg.LinAlgError:

        return (0, 0, 0)

    przesuniecie = -H_odwrotna @ g

    zinterpolowane_wartosci_ekstrem = obraz.pobierz_pixel(x, y, 0) + 0.5 * np.dot(g, przesuniecie)
    punkty_kluczowe.wartosc_ekstremum = zinterpolowane_wartosci_ekstrem

    return tuple(przesuniecie)


def punkt_na_krawedzi(punkty_kluczowe, oktawa, prog_dla_krawedzi=PROG_DLA_KRAWEDZI):
    obraz = oktawa[punkty_kluczowe.skala]
    x, y = punkty_kluczowe.i, punkty_kluczowe.j

    h11 = (
            obraz.pobierz_pixel(x + 1, y, 0)
            + obraz.pobierz_pixel(x - 1, y, 0)
            - 2 * obraz.pobierz_pixel(x, y, 0)
    )
    h22 = (
            obraz.pobierz_pixel(x, y + 1, 0)
            + obraz.pobierz_pixel(x, y - 1, 0)
            - 2 * obraz.pobierz_pixel(x, y, 0)
    )
    h12 = (
                  obraz.pobierz_pixel(x + 1, y + 1, 0)
                  - obraz.pobierz_pixel(x + 1, y - 1, 0)
                  - obraz.pobierz_pixel(x - 1, y + 1, 0)
                  + obraz.pobierz_pixel(x - 1, y - 1, 0)
          ) * 0.25

    wyznacznik_macierzy = h11 * h22 - h12 * h12
    slad_macierzy = h11 + h22

    miara = slad_macierzy ** 2 / wyznacznik_macierzy

    return miara > ((prog_dla_krawedzi + 1) ** 2 / prog_dla_krawedzi)


def znajdz_wspolrzedne_obrazu_wejsciowego(
        pkt_kluczowy,
        przesuniecie_s,
        przesuniecie_x,
        przesuniecie_y,
        sigma_min=SIGMA_MIN,
        min_piks_dyst=MIN_PIKS_DYSTANS,
        skal_na_oktawe=S_OKT,
):
    pkt_kluczowy.sigma = (
            math.pow(2, pkt_kluczowy.oktawy) * sigma_min * math.pow(2, (przesuniecie_s + pkt_kluczowy.skala) / skal_na_oktawe)
    )
    pkt_kluczowy.x = min_piks_dyst * math.pow(2, pkt_kluczowy.oktawy) * (przesuniecie_x + pkt_kluczowy.i)
    pkt_kluczowy.y = min_piks_dyst * math.pow(2, pkt_kluczowy.oktawy) * (przesuniecie_y + pkt_kluczowy.j)


def udoskonal_lub_odrzuc_punkt_kluczowy(pkt_kluczowy, oktawa, prog_kontrastu, prog_krawedzi):
    k = 0
    punkt_jest_wazny = False
    while k < MAKS_ITERACJI_UDOSKONALANIA:
        przesuniecie_s, przesuniecie_x, przesuniecie_y = aproksymacja_kwadratowa(pkt_kluczowy, oktawa, pkt_kluczowy.skala)

        maks_przesuniecie  = max(abs(przesuniecie_s), abs(przesuniecie_x), abs(przesuniecie_y ))

        pkt_kluczowy.skala += round(przesuniecie_s)
        pkt_kluczowy.i += round(przesuniecie_x)
        pkt_kluczowy.j += round(przesuniecie_y)

        if pkt_kluczowy.skala >= len(oktawa) - 1 or pkt_kluczowy.skala < 1:
            break

        poprawny_kontrast = abs(pkt_kluczowy.wartosc_ekstremum) > prog_kontrastu
        if (
                maks_przesuniecie  < 0.6
                and poprawny_kontrast
                and not punkt_na_krawedzi(pkt_kluczowy, oktawa, prog_krawedzi)
        ):
            znajdz_wspolrzedne_obrazu_wejsciowego(pkt_kluczowy, przesuniecie_s, przesuniecie_x, przesuniecie_y )
            punkt_jest_wazny  = True
            break

        k += 1

    return punkt_jest_wazny


def znajdz_punkty_kluczowe(piramida_dog, prog_kontrastu, prog_krawedzi):
    punkty_kluczowe = []
    for i in range(piramida_dog.liczba_oktaw):
        oktawa = piramida_dog.oktawy[i]

        for j in range(1, len(oktawa)):

            obraz = oktawa[j]
            for x in range(1, obraz.szerokosc - 1):
                for y in range(1, obraz.wysokosc - 1):
                    if abs(obraz.pobierz_pixel(x, y, 0)) < 0.8 * prog_kontrastu:
                        continue
                    if punkt_jest_ekstremum(oktawa, j, x, y):
                        punkt_kluczowy = PunktKluczowy(x, y, i, j, -1, -1, -1, -1)
                        punkt_jest_wazny = udoskonal_lub_odrzuc_punkt_kluczowy(
                            punkt_kluczowy, oktawa, prog_kontrastu, prog_krawedzi
                        )
                        if punkt_jest_wazny :
                            punkty_kluczowe.append(punkt_kluczowy)
    return punkty_kluczowe


def generuj_piramide_gradientu(piramida):
    piramida_gradientu = PiramidaSkal(piramida.liczba_oktaw, piramida.obrazy_na_oktawe)

    for i in range(piramida.liczba_oktaw):
        oktawa = piramida.oktawy[i]
        for obraz in oktawa:

            grad_obrazu = Obraz(szer=obraz.szerokosc, wys=obraz.wysokosc, kanal=2)

            for x in range(1, obraz.szerokosc - 1):
                for y in range(1, obraz.wysokosc - 1):
                    gx = (obraz.pobierz_pixel(x + 1, y, 0) - obraz.pobierz_pixel(x - 1, y, 0)) * 0.5
                    gy = (obraz.pobierz_pixel(x, y + 1, 0) - obraz.pobierz_pixel(x, y - 1, 0)) * 0.5

                    grad_obrazu.ustaw_pixel(x, y, 0, gx)
                    grad_obrazu.ustaw_pixel(x, y, 1, gy)

            piramida_gradientu .oktawy[i].append(grad_obrazu)

    return piramida_gradientu


def wygladz_histogram(hist):
    LICZBA_KOSZYKOW = len(hist)

    for _ in range(6):
        tymczasowy_hist = [0] * LICZBA_KOSZYKOW
        for j in range(LICZBA_KOSZYKOW):
            poprzedni_idx = (j - 1 + LICZBA_KOSZYKOW) % LICZBA_KOSZYKOW
            nastepny_idx = (j + 1) % LICZBA_KOSZYKOW
            tymczasowy_hist[j] = (hist[poprzedni_idx] + hist[j] + hist[nastepny_idx]) / 3

        hist[:] = tymczasowy_hist


def znajdz_orientacje_punktu_kluczowego(pkt_kluczowy, piramida_gradientu, lambda_orientacji, lambda_deskryptora):
    dystans_pix = MIN_PIKS_DYSTANS * math.pow(2, pkt_kluczowy.oktawy)
    obraz_gradient = piramida_gradientu.oktawy[pkt_kluczowy.oktawy][pkt_kluczowy.skala]

    min_dystans_od_granicy  = min(
        pkt_kluczowy.x, pkt_kluczowy.y, dystans_pix * obraz_gradient.szerokosc - pkt_kluczowy.x, dystans_pix * obraz_gradient.wysokosc - pkt_kluczowy.y
    )
    if min_dystans_od_granicy  <= math.sqrt(2) * lambda_deskryptora * pkt_kluczowy.sigma:
        return []

    hist = [0] * LICZBA_KOSZYKOW
    p_sigma = lambda_orientacji * pkt_kluczowy.sigma
    promien_plamy = 3 * p_sigma
    x_start = round((pkt_kluczowy.x - promien_plamy ) / dystans_pix)
    x_koniec = round((pkt_kluczowy.x + promien_plamy ) / dystans_pix)
    y_start = round((pkt_kluczowy.y - promien_plamy ) / dystans_pix)
    y_koniec = round((pkt_kluczowy.y + promien_plamy ) / dystans_pix)

    for x in range(x_start, x_koniec + 1):
        for y in range(y_start, y_koniec + 1):
            gx = obraz_gradient.pobierz_pixel(x, y, 0)
            gy = obraz_gradient.pobierz_pixel(x, y, 1)
            norma_gradientu  = math.sqrt(gx * gx + gy * gy)
            waga  = math.exp(
                -((x * dystans_pix - pkt_kluczowy.x) ** 2 + (y * dystans_pix - pkt_kluczowy.y) ** 2)
                / (2 * p_sigma ** 2)
            )
            theta = (math.atan2(gy, gx) + 2 * math.pi) % (2 * math.pi)
            koszyk = int(round(LICZBA_KOSZYKOW / (2 * math.pi) * theta)) % LICZBA_KOSZYKOW
            hist[koszyk] += waga  * norma_gradientu

    wygladz_histogram(hist)

    prog_orientacji = 0.8
    orientacje  = []
    maks_hist = max(hist)
    for j in range(LICZBA_KOSZYKOW):
        if hist[j] >= prog_orientacji * maks_hist:
            poprzedni = hist[(j - 1 + LICZBA_KOSZYKOW) % LICZBA_KOSZYKOW]
            nastepny = hist[(j + 1) % LICZBA_KOSZYKOW]
            if poprzedni > hist[j] or nastepny > hist[j]:
                continue
            theta = 2 * math.pi * (j + 1) / LICZBA_KOSZYKOW + math.pi / LICZBA_KOSZYKOW * (
                    poprzedni - nastepny
            ) / (poprzedni - 2 * hist[j] + nastepny)
            orientacje.append(theta)

    return orientacje


def aktualizuj_hist(hist, x, y, wklad, theta_mn, lambda_deskryptor):
    for i in range(1, LICZBA_HIST + 1):
        x_i = (i - (1 + LICZBA_HIST) / 2) * 2 * lambda_deskryptor / LICZBA_HIST
        if abs(x_i - x) > 2 * lambda_deskryptor / LICZBA_HIST:
            continue
        for j in range(1, LICZBA_HIST + 1):
            y_j = (j - (1 + LICZBA_HIST) / 2) * 2 * lambda_deskryptor / LICZBA_HIST
            if abs(y_j - y) > 2 * lambda_deskryptor / LICZBA_HIST:
                continue

            waga_histogramu  = (1 - LICZBA_HIST * 0.5 / lambda_deskryptor * abs(x_i - x)) * (
                    1 - LICZBA_HIST * 0.5 / lambda_deskryptor * abs(y_j - y)
            )

            for k in range(1, LICZBA_ORIENTACJI + 1):
                theta_k = 2 * math.pi * (k - 1) / LICZBA_ORIENTACJI
                roznica_theta = (theta_k - theta_mn + 2 * math.pi) % (2 * math.pi)
                if abs(roznica_theta) >= 2 * math.pi / LICZBA_ORIENTACJI:
                    continue
                waga_koszyka = 1 - LICZBA_ORIENTACJI * 0.5 / math.pi * abs(roznica_theta)
                hist[i - 1][j - 1][k - 1] += waga_histogramu  * waga_koszyka * wklad


def histogramy_na_wektor(histogramy):
    histogram_splaszczony = np.array(histogramy).flatten()

    norm = np.linalg.norm(histogram_splaszczony)

    histogram_splaszczony = np.minimum(histogram_splaszczony, 0.2 * norm)
    norm2 = np.linalg.norm(histogram_splaszczony)

    wektor_cech = np.floor(512 * histogram_splaszczony / norm2).clip(0, 255).astype(np.uint8)

    return wektor_cech


def oblicz_punktykluczowe_deskryptory(pk, theta, grad_pyramid, lambda_deksryptor):
    dystans_piks = MIN_PIKS_DYSTANS * math.pow(2, pk.oktawy)
    gradienT_obrazu = grad_pyramid.oktawy[pk.oktawy][pk.skala]
    histogramy = [
        [[0 for _ in range(LICZBA_ORIENTACJI)] for _ in range(LICZBA_HIST)] for _ in range(LICZBA_HIST)
    ]

    polowa_rozmiaru = math.sqrt(2) * lambda_deksryptor * pk.sigma * (LICZBA_HIST + 1) / LICZBA_HIST
    x_start = round((pk.x - polowa_rozmiaru) / dystans_piks)
    x_koniec = round((pk.x + polowa_rozmiaru) / dystans_piks)
    y_start = round((pk.y - polowa_rozmiaru) / dystans_piks)
    y_koniec = round((pk.y + polowa_rozmiaru) / dystans_piks)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    sigma = lambda_deksryptor * pk.sigma

    for m in range(x_start, x_koniec + 1):
        for n in range(y_start, y_koniec + 1):

            x = (
                        (m * dystans_piks - pk.x) * cos_t + (n * dystans_piks - pk.y) * sin_t
                ) / pk.sigma
            y = (
                        -(m * dystans_piks - pk.x) * sin_t + (n * dystans_piks - pk.y) * cos_t
                ) / pk.sigma

            if max(abs(x), abs(y)) > lambda_deksryptor * (LICZBA_HIST + 1) / LICZBA_HIST:
                continue

            gx = gradienT_obrazu.pobierz_pixel(m, n, 0)
            gy = gradienT_obrazu.pobierz_pixel(m, n, 1)
            theta_mn = (math.atan2(gy, gx) - theta + 4 * math.pi) % (2 * math.pi)
            norma_gradientu  = math.sqrt(gx ** 2 + gy ** 2)
            wagi = math.exp(
                -((m * dystans_piks - pk.x) ** 2 + (n * dystans_piks - pk.y) ** 2)
                / (2 * sigma ** 2)
            )
            wartosc = wagi * norma_gradientu

            aktualizuj_hist(histogramy, x, y, wartosc, theta_mn, lambda_deksryptor)

    pk.deskryptor = histogramy_na_wektor(histogramy)


def znajdz_punkty_kluczowe_i_deskryptory(
        obraz,
        sigma_min=SIGMA_MIN,
        liczba_oktaw=L_OKT,
        sklana_na_oktawe=S_OKT,
        prog_kontr=KONTRAST_DOG,
        prog_krawedzi=PROG_DLA_KRAWEDZI,
        lambda_orient=LAMBDA_ORIENTACJA,
        lambda_desk=LAMBDA_DEKSRYPTOR,
):
    assert obraz.liczba_kanalow == 1 or obraz.liczba_kanalow == 3

    wejsciowy_obraz = obraz if obraz.liczba_kanalow == 1 else obraz_rgb_do_szarosci(obraz)
    piramida_gauss = tworzenie_piramidy(wejsciowy_obraz)
    piramida_dog = tworzenie_piramidy_dog(piramida_gauss)
    tymczasowe_punkty_kluczowe = znajdz_punkty_kluczowe(piramida_dog, prog_kontr, prog_krawedzi)
    piramida_gradientu = generuj_piramide_gradientu(piramida_gauss)

    PK = []

    for PK_TYMCZASOWE in tymczasowe_punkty_kluczowe:
        orientacje = znajdz_orientacje_punktu_kluczowego(
            PK_TYMCZASOWE, piramida_gradientu, lambda_orient, lambda_desk
        )
        for theta in orientacje:
            _pk = PK_TYMCZASOWE
            oblicz_punktykluczowe_deskryptory(_pk, theta, piramida_gradientu, lambda_desk)
            PK.append(_pk)

    return PK


def odleglosc_euklidesowa(a, b):
    roznica = np.array(a, dtype=int) - np.array(b, dtype=int)
    odleglosc = np.sum(roznica ** 2)
    return np.sqrt(odleglosc)


def znajdz_dopasowania_kluczy_punktowych(
        a, b, p_relat=PROG_RELATYWNY, p_abs=PROG_ABSOLUTNY
):
    dopasowania = []

    for i, ka in enumerate(a):
        nn1_idx = -1
        nn1_dyst = 100000000
        nn2_dyst = 100000000

        for j, kb in enumerate(b):
            dyst = odleglosc_euklidesowa(ka.deskryptor, kb.deskryptor)
            if dyst < nn1_dyst:
                nn2_dyst = nn1_dyst
                nn1_dyst = dyst
                nn1_idx = j
            elif nn1_dyst <= dyst < nn2_dyst:
                nn2_dyst = dyst

        if nn1_dyst < p_relat * nn2_dyst and nn1_dyst < p_abs:
            dopasowania.append((i, nn1_idx))

    return dopasowania


def rysuj_punkty_kluczowe(ob, punkty_kluczowe):
    obraz = ob.kopiuj_obraz()

    for punkt_kluczowy in punkty_kluczowe:
        cv2.circle(obraz.piksele, (int(punkt_kluczowy.x), int(punkt_kluczowy.y)), 5, (0, 0, 255), -1)

    return obraz


def hsv_na_rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def rysuj_dopasowania(a: Obraz, b, punkty_kluczowe_A, punkty_kluczowe_B, dopasowania):
    wys = max(a.wysokosc, b.wysokosc)
    szer = a.szerokosc + b.szerokosc

    obraz_wynikowy = Obraz(szer=szer, wys=wys, kanal=3)

    obraz_wynikowy.piksele[: a.wysokosc, : a.szerokosc] = a.piksele
    obraz_wynikowy.piksele[: b.wysokosc, a.szerokosc: a.szerokosc + b.szerokosc] = b.piksele

    for i, m in enumerate(dopasowania):

        PK_A = punkty_kluczowe_A[m[0]]
        PK_B = punkty_kluczowe_B[m[1]]
        punkt_A = (int(PK_A.x), int(PK_A.y))
        punkt_B = (int(PK_B.x) + a.szerokosc, int(PK_B.y))

        if punkt_B[0] < punkt_A[0]:
            punkt_A, punkt_B = punkt_B, punkt_A
        hue = (i / len(dopasowania)) % 1
        kolor = hsv_na_rgb(hue, 1, 1)

        cv2.line(obraz_wynikowy.piksele, punkt_A, punkt_B, kolor, 1)

    return obraz_wynikowy

# python program.py a.jpg b.jpg
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match keypoints between two images.")
    parser.add_argument("image1", help="First image file")
    parser.add_argument("image2", help="Second image file")
    args = parser.parse_args()

    print("Prosze czekac, program rozpoczal dzialanie")
    obraz1 = Obraz(args.image1)
    szary_obraz1 = obraz_rgb_do_szarosci(obraz1)
    klucze_punktowe1 = znajdz_punkty_kluczowe_i_deskryptory(szary_obraz1)

    obraz2 = Obraz(args.image2)
    szary_obraz2 = obraz_rgb_do_szarosci(obraz2)
    klucze_punktowe2 = znajdz_punkty_kluczowe_i_deskryptory(szary_obraz2)

    dopasowania = znajdz_dopasowania_kluczy_punktowych(klucze_punktowe1, klucze_punktowe2)

    result = rysuj_dopasowania(obraz1, obraz2, klucze_punktowe1, klucze_punktowe2, dopasowania)
    result.zapisz_do_pliku("rezultat.jpg")
    print("Obraz wynikowy został zapisany do rezultat.jpg")