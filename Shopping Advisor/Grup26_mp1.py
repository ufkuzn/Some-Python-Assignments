import json
import dbm

urundb = dbm.open('Urun.db', 'c')
siparisdb = dbm.open('siparis.db', 'c')
kullanicidb = dbm.open('Kullanici.db', 'c')

def urunleri_oku(dosya_ismi):
    file = open(dosya_ismi, 'r')
    urunler = json.load(file)
    count = 1
    for i in urunler:
        liste = []
        liste.append(i['isim'])
        urundb ['urun-{0:04}'.format(count)] = str(i['isim'])
        count += 1
    #print(urundb['urun-0001'])
    file.close()
    urundb.close()
    return urunler

class Urun():
    def __init__(self, isim, tanim, marka, link, fiyat, kategori, stok):
        self.isim = isim
        self.tanim = tanim
        self.marka = marka
        self.link = link
        self.fiyat = fiyat
        self.kategori = kategori
        self.stok = stok
        
        if type(fiyat)==str or type(stok)==str:
            try:
                self.fiyat = float(fiyat)
                self.stok = int(stok)
            except:
                raise Exception("Fiyat veya stok degiskenleri float/int yapisinda veya numerik string yapisinda olmalidir")

    def fiyat_arttir(self, artis_miktari):
        self.fiyat += artis_miktari
    
    def kampanya(self, yuzde):
        ''' Yapilan kampanya yuzdesi oraninda indirim yapar.'''
        self.fiyat *= (1-yuzde)

    def __str__(self):
        return "{} : urun ismi {}, kategorisi {}, fiyati {}, kalan adet {}".format(self.isim, self.kategori, self.fiyat, self.stok)


class Siparis():
    def __init__(self, kimlik):
        self.kimlik = kimlik
        

class Kullanici():
    def __init__(self, kullaniciIsmi, adet):
        self.kullaniciIsmi = kullaniciIsmi
        self.adet = adet
        self.toplamHarcama = 0.0
        self.siparisListesi = ['Selpak Tuvalet Kagidi']
        kullanicidb ['Kullanici-0001'] = self.kullaniciIsmi
        siparisdb ['Siparis-0001'] = self.siparisListesi[0]
    
    def harcama_miktari(self, urun:Urun):
        if(urun.stok>0):
            urun.stok -= self.adet
            self.siparisListesi.append(urun1)
            self.toplamHarcama += urun.fiyat * self.adet
        else:
            pass
        return self.toplamHarcama

urun_kodlari = int(input("Lutfen bir urun kodu seciniz [1-17]: "))
satin_alma_adedi = int(input("Lutfen satin alma adedi giriniz: "))
baska_urun = input("Baska bir urun girmek ister misiniz? [E/H]: ")
if (baska_urun == 'E' or baska_urun == 'e'):
    pass
else:
    pass
    
urun1 = Urun('Selpak Tuvalet Kagidi', "temiz", "Selpak", "https://cdn.getir.com/product/55a004bf35a77b0c0075cd0c_tr_1602362018887.jpeg", 29.90, ["Ev Bakim", "KagitUrunleri"], 10)
kullanici1 = Kullanici("Ufuk", 2)
siparis1 = Siparis(kullanicidb["Kullanici-0001"].decode())

harcama = kullanici1.harcama_miktari(urun1)
print("{} toplamda {} adet siparis vermis ve {} TL harcamistir.".format('Kullanici-0001', satin_alma_adedi, harcama))

urunleri_oku("urunler_temiz.json")
siparisdb.close()
kullanicidb.close()
