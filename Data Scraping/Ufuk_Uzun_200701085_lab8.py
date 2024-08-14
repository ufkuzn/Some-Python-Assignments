import urllib.request as urllib2
from bs4 import BeautifulSoup
import re
import shelve


class WebScrapper:
    def __init__(self, URL, db):
        self.URL = URL
        self.db = db
        self.get_categories(self.URL)

    def get_soup(self, webpage):
        request = urllib2.Request(webpage)
        c = urllib2.urlopen(request)
        soup = BeautifulSoup(c.read(), 'html.parser')
        return soup

    def get_categories(self, webpage):
        soup = self.get_soup(webpage)
        urls = [self.URL[:-10] + x.get('href') for x in soup.find_all("a", href=re.compile("catalogue/category/books"))]
        urls = urls[1:]

        categories = []
        for url in urls:
            categories.append(url[51:-14])

        get_categories = {}
        for i in range(len(categories)):
            get_categories[categories[i]] = urls[i]
        print(get_categories)

        all_cate_book = []
        for i in get_categories:
            main_url = get_categories[i]
            soup = self.get_soup(main_url)
            page_urls = ["https://books.toscrape.com/catalogue/" + x.div.a.get('href')[9:] for x in soup.findAll("article", class_="product_pod")]
            all_cate_book.append(page_urls)
        print(all_cate_book)

    def create_db(self, returning_data):
        shfile = shelve.open(self.db)
        for i in returning_data:
            shfile[i] = returning_data[i]
        shfile.close()
