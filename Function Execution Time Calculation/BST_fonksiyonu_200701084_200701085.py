import os
import datetime
import time

class BST:
    def __init__(self):
        self.root = None
    class Node:
        def __init__(self, key, value, n):
            self.left = None
            self.right = None
            self.key = key
            self.val = value
            self.n = n
    def size(self):
        return self._size(self.root)
    def _size(self, x):
        if x == None:
            return 0
        else:
            return x.n

    start_time_get = time.time()
    def get(self, key):
        return self._get(self.root, key)

    def _get(self, x, key):
        if x == None:
            return None
        if key < x.key:
            return self._get(x.left, key)
        elif key > x.key:
            return self._get(x.right, key)
        else:
            return x.val

    def put(self, key, val):
        self.root = self._put(self.root, key, val)

    def _put(self, x, key, val):
        if x == None:
            return BST.Node(key, val, 1)
        if key < x.key:
            x.left = self._put(x.left, key, val)
        elif key > x.key:
             x.right = self._put(x.right, key, val)
        else:
            x.val = val
        x.n = self._size(x.left) + self._size(x.right) + 1
        return x






def main():
    symbolTable = BST()


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

    file.close()
    '''VERİLEN TEXT DOSYASINI OKUMAYI BİTİRİR.'''

    tumu = {}
    for word in kelimeler:
        if word not in tumu:
            tumu[word] = 1
        else:
            tumu[word] += 1

    startput = datetime.datetime.now()
    i = 0
    for key in tumu.keys():
        symbolTable.put(key, i)  # DEĞERLER TABLOYA PUT EDİLİR. HER BİR DEGER SADECE BİR KEZ TABLOYA YERLEŞTİRİLİR.
        i += 1
    endput = datetime.datetime.now()

    startolustur  = datetime.datetime.now()
    root = symbolTable.root
    traverse(root)
    endolustur  = datetime.datetime.now()

    print()
    print("*" * 100)

    '''ORTALAMA PUT SÜRESİNİ MS OLARAK DÖNDÜRÜR.'''
    print()
    elapsedput = endput - startput
    print("Ortalama Put Suresi = ", elapsedput.seconds, ":", elapsedput.microseconds, "ms")


    '''TABLONUN OLUŞTURULMA SÜRESİNİ MS OLARAK DÖNDÜRÜR.'''
    print()
    elapsed_time = endolustur  - startolustur
    print("Sembol Tablosu Olusturma = ", elapsed_time.seconds, ":", elapsed_time.microseconds, "ms")

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

def traverse(root):
    if root.left != None:
        traverse(root.left)
    print (root.key, " ", root.val)
    if root.right != None:
        traverse(root.right)


if __name__ == '__main__':
 main()


