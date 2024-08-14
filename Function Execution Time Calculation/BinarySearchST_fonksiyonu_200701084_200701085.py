import os
import datetime
import time


class BinarySearchST():
    def __init__(self):
        self.keys = []
        self.values = []
        self.n = 0

    def size(self):
        return self.n

    start_time_get = time.time()
    def get (self, key):
        i = self.binarySearch(key)
        if i < self.n and self.keys[i] == key:
            return self.values[i]
        else:
            return None


    def put (self, key, value):
        i = self.binarySearch(key)
        if i < self.n and self.keys[i] == key:
            self.values[i] = value
            return
        self.keys.append(' ')
        self.values.append(' ')
        for j in range (self.n, i, -1):
            self.keys[j] = self.keys[j-1]
            self.values[j] = self.values[j - 1]
        self.keys[i] = key
        self.values[i] = value
        self.n += 1


    def binarySearch(self, key):
        lo = 0
        hi = self.n - 1
        while (lo <= hi):
            mid = lo + (hi - lo) // 2
            if key < self.keys[mid]:
                hi = mid - 1
            elif key > self.keys[mid]:
                lo = mid + 1
            else:
                return mid
        return lo


def main():
    symbolTable = BinarySearchST()

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


    startolustur   = datetime.datetime.now()
    for i in range(symbolTable.n-1, -1, -1):
        print (symbolTable.keys[i], "  ", symbolTable.values[i])
    endolustur   = datetime.datetime.now()


    print()
    print("*"*100)
    '''ORTALAMA PUT SÜRESİNİ MS OLARAK DÖNDÜRÜR.'''
    print()
    elapsedput = endput - startput
    print("Ortalama Put Suresi = ", elapsedput.seconds, ":", elapsedput.microseconds, "ms")

    '''TABLONUN OLUŞTURULMA SÜRESİNİ MS OLARAK DÖNDÜRÜR.'''
    print()
    elapsed_time = endolustur  - startolustur
    print("Sembol Tablosu Olusturma = ", elapsed_time, "ms")

    '''TABLODA VAR OLAN(HİT) BİR ANAHTARIN GET SÜRESİNİ MS OLARAK DÖNDÜRÜR'''
    print()
    print("'{}' Kelimesinin Tablodaki karşılığı = {}".format("temper", symbolTable.get('temper')))
    elapsed_time = time.time() - symbolTable.start_time_get
    print("Varolan Anahtarın Ortalama Get Suresi = ", elapsed_time, "ms")
    print()

    '''TABLODA VAR OLMAYAN(MİSS) BİR ANAHTARIN GET SÜRESİNİ MS OLARAK DÖNDÜRÜR'''
    print("'{}' Kelimesinin Tablodaki karşılığı = {}".format("selam", symbolTable.get('selam')))
    elapsed_time = time.time() - symbolTable.start_time_get
    print("Varolmayan Anahtarın Ortalama Get Suresi = ", elapsed_time, "ms")
    print()




if __name__ == '__main__':
    main()
