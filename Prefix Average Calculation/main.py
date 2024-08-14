import random
import time

n = 1     #BURADAN FONKSİYONLARIN GİRDİ BOYUTLARI SEÇİLİR.. (1, 10, 100, 1000, 10000,100000 ve 1000000)
S = []
for i in range(n):
    S.append(random.randint(0,100))
#A = []
print()

def prefix_average(S):
    A = []
    start0 = time.time()
    for j in range(n-1):
        total = 0
        for i in range(j):
            total = total + S[i]
        A.append(total/(j+1))
    #print(A)                      #BURADAN YENİ A DİZİNİNE ERİŞEBİLİRSİNİZ.
    end0 = time.time()
    print("1. FONKSİYONUN ZAMANI=",format(end0 - start0))
    print()

def prefix_average_with_sum (S):
    A = []
    start1 = time.time()
    for j in range(n-1):
        A.append(sum(S)/(j+1))
   # print(A)                        #BURADAN YENİ A DİZİNİNE ERİŞEBİLİRSİNİZ.
    end1 = time.time()
    print("2. FONKSİYONUN ZAMANI=", format(end1 - start1))
    print()


def prefix_average_with_running_average(S):
    A = []
    start2 = time.time()
    total1 = 0
    for j in range (n-1):
        total1 = total1+S[j]
        A.append(total1/(j+1))
   # print(A)     #BURADAN YENİ A DİZİNİNE ERİŞEBİLİRSİNİZ.
    end2 = time.time()
    print("3. FONKSİYONUN ZAMANI=",format(end2 - start2))
    print()

if __name__ == '__main__':
    prefix_average(S)
    prefix_average_with_sum(S)
    prefix_average_with_running_average(S)