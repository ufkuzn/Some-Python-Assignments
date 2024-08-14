from Ufuk_Uzun_200701085_lab8 import *


def __main__():
    db_name = 'kitaplar.db'
    BASE_URL = "http://books.toscrape.com/index.html"
    ws = WebScrapper(BASE_URL, db_name)


if __name__ == '__main__':
    __main__()