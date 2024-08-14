import os
import datetime
import time
import matplotlib.pyplot as plt

class seperateChainingHashST():
    def __init__(self, M):
        self.M = M
        self.st = []
        for i in range(M):
            self.st.append(None)

    class Node:
        def __init__(self, key, value, next):
            self.key = key
            self.val = value
            self.next = next

    def hash(self, key):
        return (hash(key) & 0x7fffffff) % self.M

    start_time_get = time.time()
    def get(self, key):
        i = self.hash(key)
        x = self.st[i]
        while x != None:
            if key == x.key:
                return x.val
            x = x.next
        return None

    def put (self, key, val):
        i = self.hash(key)
        x = self.st[i]
        while x != None:
            if key == x.key:
                x.val = val
                return
            x = x.next
        self.st[i] = seperateChainingHashST.Node(key, val, self.st[i])


def main():
    symbolTable = seperateChainingHashST(97)

    '''VERİLEN TEXT DOSYASINI OKUR'''

    yol = os.path.dirname(os.path.abspath(__file__))
    dosya = os.path.join(yol, 'kitap.txt')
    file = open(dosya)

    satir = file.read()

    ayirma = satir.split()

    kelimeler = list()
    for i in ayirma:
        i = i.strip(' ')
        i = i.strip('.')
        i = i.strip('\n')
        i = i.strip(',')
        i = i.strip('"')
        i = i.strip('_')
        i = i.strip(':')
        i = i.strip('[')
        i = i.strip(']')
        i = i.strip(';')
        kelimeler.append(i.lower())
    # print(kelimeler)

    file.close()
    '''VERİLEN TEXT DOSYASINI OKUMAYI BİTİRİR.'''

    tumu = {}
    for word in kelimeler:
        if word not in tumu:
            tumu[word] = 1
        else:
            tumu[word] += 1
    # print(tumu)

    startput = datetime.datetime.now()
    i = 0
    for key in tumu.keys():
        symbolTable.put(key, i)  # DEĞERLER TABLOYA PUT EDİLİR. HER BİR DEGER SADECE BİR KEZ TABLOYA YERLEŞTİRİLİR.
        i += 1
    endput = datetime.datetime.now()



    '''BU BOLUMDE GOZLERIN OLUSTURULMASININ YANINDA ARTI OLARAK BIR BOS LISTE TANIMLANIP BU LISTENIN ICERISINE HANGI GOZDE KAC TANE CARPISMA OLDUGU YAZDIRILDI.'''

    list1 = {}
    counter = 1
    startolustur = datetime.datetime.now()
    for i in range(symbolTable.M):
        x = symbolTable.st[i]
        print ("Goz:{}".format(i), end="::::")
        while  x != None:
            print (x.key, "  ", x.val, end="--->     ")
            x = x.next
            counter += 1
        list1["Goz {}".format(i)] = counter
        counter = 1
        print()
    endolustur = datetime.datetime.now()


    print("*" * 100)
    print()
   # print("ÇARPIŞMA SAYISI LİSTEM = {}".format(list1))     #HANGİ GÖZDE NE KADAR ÇARPIŞMA OLDUĞUNU YAZDIRIR.

    print("*" * 100)

    print()
    '''ORTALAMA PUT SURESINI MS OLARAK DONDURUR.'''
    print()
    elapsedput = endput - startput
    print("Ortalama Put Suresi = ", elapsedput , "ms")

    '''TABLONUN OLUSTURULMA SURESINI MS OLARAK DONDURUR'''
    print()
    elapsed_time = endolustur - startolustur
    print("Sembol Tablosu Olusturma = ", elapsed_time, "ms")

    '''TABLONUN VAR OLAN(HIT) BIR ANAHTARIN GET SURESINI MS OLARAK DONDURUR.'''
    print()
    print("'{}' Kelimesinin Tablodaki karşılığı = {}".format("alive", symbolTable.get('alive')))
    elapsed_time = time.time() - symbolTable.start_time_get
    print("Varolan Anahtarın Ortalama Get Suresi = ", elapsed_time, "ms")
    print()

    '''TABLODA VAR OLMAYAN(MISS) BIR ANAHTARIN GET SURESINI MS OLARAK DONDURUR.'''
    print("'{}' Kelimesinin Tablodaki karşılığı = {}".format("selam", symbolTable.get('selam')))
    elapsed_time = time.time() - symbolTable.start_time_get
    print("Varolmayan Anahtarın Ortalama Get Suresi = ", elapsed_time, "ms")
    print()



    '''BURADA GRAFİK OLUŞTURULUR'''
    '''GRAFİK İÇİN MATPLOTLİB KÜTÜPHANESİ KULLANILDI. '''
    '''AÇILAN PENCEREDEN ÇIKIŞ YAPMAK İÇİN ALT+F4 KOMBİNASYONUNU KULLANINIZ.'''


    x = list1.keys()
    y = list1.values()


    fig, ax = plt.subplots()

    ax.bar(x, y, width=1, edgecolor="red", linewidth=0.7,label="gözler")

    ax.set(xlim=(0, 97))
    plt.xticks(rotation=90)


    plt.ylabel("CARPISMA SAYISI")
    plt.xlabel("GOZLER")

    plt.legend()
    plt.title("(CIKIS ICIN : ALT+F4)\n"
              "GOZLERDEKI CARPISMA SAYISI")
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()



if __name__ == '__main__':
    main()
