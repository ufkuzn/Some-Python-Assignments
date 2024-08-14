# Ogrenci no: 200701085
# Ad: UFUK
# Soyad: UZUN
# Ders Subesi: 1
# Odev no: 6
def kriptolama(metin, n):
    kriptolanmis_metin = ""

    for i in range(len(metin)):
        harf = metin[i]
        if harf.isupper():
            kriptolanmis_metin += chr((ord(harf) + int(n) - 65) % 26 + 65)
        else:
            kriptolanmis_metin += chr((ord(harf) + int(n) - 97) % 26 + 97)
    print("Kriptolanmis metin: ", kriptolanmis_metin)


metin = input("Kriptolanacak metni giriniz: ")
n = input("Girilen harften kac harf sonrasi kripto harf olacak?: ")
kriptolama(metin, n)


def kripto_cozme(metin, n):
    cozulecek_metin = ""

    for i in range(len(metin)):
        harf = metin[i]
        if harf.isupper():
            cozulecek_metin += chr((ord(harf) - int(n) - 65) % 26 + 65)
        else:
            cozulecek_metin += chr((ord(harf) - int(n) - 97) % 26 + 97)
    print("Kriptosu cozulen metin: ", cozulecek_metin)


metin = input("Kriptosu cozulecek metni giriniz: ")
kripto_cozme(metin, n)
