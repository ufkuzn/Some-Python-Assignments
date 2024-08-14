import os
import datetime
import time

class SequentialST:
    def __init__(self):
        self.first = None
        self.n = 0

    class Node:
        def __init__(self, key, value, next):
            self.key = key
            self.value = value
            self.next = next

    start_time = time.time()
    def get(self, key):

        x = self.first
        while x != None:
            if x.key == key:
                return x.value
            x = x.next
        return None

    def put(self, key, value):
        x = self.first
        while x != None:
            if x.key == key:
                x.value = value
                return
            x = x.next
        self.first = SequentialST.Node(key, value, self.first)
        self.n += 1

    def keys(self):
        ll = []
        x = self.first
        while x != None:
            ll.append(x.key)
            x = x.next
        return ll



def main():
    symbolTable = SequentialST()


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
    # print(tümü)


    startput = datetime.datetime.now()
    i = 0
    for key in tumu.keys():
        symbolTable.put(key, i)  #DEĞERLER TABLOYA PUT EDİLİR. HER BİR DEGER SADECE BİR KEZ TABLOYA YERLEŞTİRİLİR.
        i += 1
    endput = datetime.datetime.now()


    starttabloolusturma = datetime.datetime.now()
    head = symbolTable.first
    while head != None:
        print(head.key, "  ", head.value)
        head = head.next
    endtabloolusturma = datetime.datetime.now()


    print()
    print("*" * 100)
    '''OLUŞTURULAN TABLONUN ANAHTARLARINI BASAR'''
    print()
   # print("TABLONUN ANAHTARLARI = ",symbolTable.keys())


    '''ORTALAMA PUT SÜRESİNİ MS OLARAK DÖNDÜRÜR.'''
    print()
    elapsedput = endput - startput
    print("Ortalama Put Suresi = ", elapsedput.seconds,":",elapsedput.microseconds,"ms")


    '''TABLONUN OLUŞTURULMA SÜRESİNİ MS OLARAK DÖNDÜRÜR.'''
    print()
    elapsedtabloolusturma = endtabloolusturma - starttabloolusturma
    print("Sembol Tablosu Olusturma = ",elapsedtabloolusturma.seconds, ":", elapsedtabloolusturma.microseconds,"ms")


    '''TABLODA VAR OLAN(HİT) BİR ANAHTARIN GET SÜRESİNİ MS OLARAK DÖNDÜRÜR'''
    print()
    print("'{}' Kelimesinin Tablodaki karşılığı = {}".format("temper",symbolTable.get('temper')))
    elapsed_time = time.time() - symbolTable.start_time
    print("Varolan Anahtarın Ortalama Get Suresi = ", elapsed_time,"ms")
    print()


    '''TABLODA VAR OLMAYAN(MİSS) BİR ANAHTARIN GET SÜRESİNİ MS OLARAK DÖNDÜRÜR'''
    print()
    print("'{}' Kelimesinin Tablodaki karşılığı = {}".format("selam", symbolTable.get('selam')))
    elapsed_time = time.time() - symbolTable.start_time
    print("Varolmayan Anahtarın Ortalama Get Suresi = ", elapsed_time, "ms")
    print()


if __name__ == '__main__':
    main()
