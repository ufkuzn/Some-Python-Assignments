import re
import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("Mobile phone price.csv")

print("Veri setindeki her sütunun ilk 5 satırı \n", df.head(), '\n\n' + '*' * 70)

for column in df.columns:
    col = df[column]
    unique_value = col.unique()
    col_type = col.dtype
    null_count = col.isnull().sum()
    print(' ' * 10, column)
    if col_type == 'object':
        desc = col.describe()
        print(f"Değer (sütun) sayısı: {desc['count']}")
        print(f"Benzersiz değerler sayısı: {desc['unique']}")
        print(f"En çok tekrar eden : {desc['top']}")
        print(f"En çok tekrar eden sayısı: {desc['freq']}")
    elif col_type == 'int64':
        desc = col.describe()
        print(f"Değer (sütun) sayısı: {desc['count']}")
        print(f"Ortalama: {desc['mean']}")
        print(f"Standart sapma: {desc['std']}")
        print(f"Minimum değer: {desc['min']}")
        print(f"Çeyreklikler: 25%: {desc['25%']}, 50%: {desc['50%']}, 75%: {desc['75%']}")
        print(f"Maksimum değer: {desc['max']}")
    print(f"Toplam eksik değer sayısı: {null_count}")
    print(f"Benzersiz değerler: {unique_value}")
    print('\n' + '*' * 70)

df = df.rename(columns=lambda x: x.strip())


def extract_capacity(input_list):
    capacity_list = []
    for item in input_list:
        size = re.findall(r'\d+', item)
        if size:
            capacity_list.append(size[0])
    return capacity_list


def get_numeric_screen_sizes(sizes):
    numeric_sizes = []
    for size in sizes:
        try:
            num = float(size)
            if num.is_integer():
                num = int(num)
            numeric_sizes.append(str(num))
        except ValueError:
            parts = re.findall(r'\d+.\d+|\d+', size)
            total = sum(float(part) for part in parts)
            if total.is_integer():
                total = int(total)
            numeric_sizes.append(str(total))
    return numeric_sizes


def format_camera_specs(camera_specs):
    total_camera_specs = 0
    for csv_string in camera_specs:
        parts = csv_string.split("+")
        for part in parts:
            part = part.strip()
            if part.endswith("MP"):
                part = part[:-2]
                try:
                    total_camera_specs += int(part)
                except ValueError:
                    pass
            elif "3D" in part or "ToF" in part:
                pass
            else:
                part = part[:-3]
                try:
                    total_camera_specs += int(part)
                except ValueError:
                    pass
    return str(total_camera_specs)


def get_numeric_prices(prices):
    numeric_prices = []
    for price in prices:
        price = re.sub('[^\d]', '', price)
        numeric_prices.append(int(price))
    return numeric_prices

class MobilePhone:
    def __init__(self, data_name):
        self.data_name = data_name
        self.original_data = None
        if self.original_data is None:
            self.upload()
        # self.firstAnalysis()
        self.clean_data = None
        self.clean()


    def upload(self):
        self.original_data = pd.read_csv(self.data_name)


    def firstAnalysis(self):
        data = self.original_data
        print("Veri setindeki her sütunun ilk 5 satırı \n", data.head(), '\n\n' + '*' * 70)
        for column in data.columns:
            col = data[column]
            unique_value = col.unique()
            col_type = col.dtype
            null_count = col.isnull().sum()
            print(' ' * 10, column)
            if col_type == 'object':
                desc = col.describe()
                print(f"Değer (sütun) sayısı: {desc['count']}")
                print(f"Benzersiz değerler sayısı: {desc['unique']}")
                print(f"En çok tekrar eden : {desc['top']}")
                print(f"En çok tekrar eden sayısı: {desc['freq']}")
            elif col_type == 'int64':
                desc = col.describe()
                print(f"Değer (sütun) sayısı: {desc['count']}")
                print(f"Ortalama: {desc['mean']}")
                print(f"Standart sapma: {desc['std']}")
                print(f"Minimum değer: {desc['min']}")
                print(f"Çeyreklikler: 25%: {desc['25%']}, 50%: {desc['50%']}, 75%: {desc['75%']}")
                print(f"Maksimum değer: {desc['max']}")
            print(f"Toplam eksik değer sayısı: {null_count}")
            print(f"Benzersiz değerler: {unique_value}")
            print('\n' + '*' * 70)


    def clean(self):
        data = self.original_data.copy()
        data.columns = [col.strip() for col in data.columns]
        data['Brand'] = data['Brand']
        data['Model'] = data['Model']
        data['Storage'] = extract_capacity(data['Storage'])
        data['RAM'] = extract_capacity(data['RAM'])
        data['Screen Size (inches)'] = get_numeric_screen_sizes(data['Screen Size (inches)'])
        data['Camera (MP)'] = format_camera_specs(data['Camera (MP)'])
        data['Battery Capacity (mAh)'] = data['Battery Capacity (mAh)']
        data['Price ($)'] = get_numeric_prices(data['Price ($)'])
        self.clean_data = data
        print(self.clean_data.to_string(index=False))

    def train_linear_regression(self, X_col, y_col):
        X = self.clean_data[X_col].values.reshape(-1, 1)
        y = self.clean_data[y_col]
        lr = LinearRegression()
        lr.fit(X, y)
        print("\n**********************************************************************\n")
        print(f"R-kare değeri: {lr.score(X, y):.4f}")
        print(f"Kesme: {lr.intercept_:.4f}")
        print(f"Eğim: {lr.coef_[0]:.4f}")
        return lr

    def predict_price(self, verilen, istenen, verilen_deger):
        lr_model = self.train_linear_regression(verilen, istenen)
        price = lr_model.predict([[verilen_deger]])
        return price[0]


def main():
    data_name = "Mobile phone price.csv"
    phone = MobilePhone(data_name)
    verilen = 'Battery Capacity (mAh)'
    istenen = 'Price ($)'
    verilen_deger = 1000
    predicted_price = phone.predict_price(verilen, istenen, verilen_deger)
    print(f"Tahmini fiyat: ${predicted_price:.2f}")


if __name__ == '__main__':
    main()