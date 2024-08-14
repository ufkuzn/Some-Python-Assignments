import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


class DataAnalyzer:
    def __init__(self, df):
        self.df = df

    def get_columns(self):
        return list(self.df.columns)

    def print_separator(self):
        # Ayırıcı çizgisi yazdırma
        print('\n' + '*' * 70)

    def unique_values(self, column):
        # Sütundaki benzersiz değerleri yazdırma
        return self.df[column].unique()

    def check_missing_values(self):
        # Eksik değer kontrolü yapma ve hatalı sütunları döndürme
        return [column for column in self.df.columns if self.df[column].isnull().sum() > 0]

    def get_column_dtype(self, column):
        # Sütunun veri tipini döndürme
        return self.df[column].dtype

    def get_object_columns(self):
        # Object veri türüne sahip sütunları döndürme
        return [column for column in self.df.columns if self.get_column_dtype(column) == 'object']

    def print_column_info(self, column):
        # Sütun hakkında bilgileri yazdırma
        column_values = self.df[column]
        print(f"Sütun adı: {column}")
        print(f"Veri türü: {self.get_column_dtype(column)}")
        print(f"Toplam eksik değer sayısı: {column_values.isnull().sum()}")

    def print_error_columns(self, error_columns):
        # Hatalı sütunları yazdırma
        if error_columns:
            print("Hatalı sütun bulundu!\nHatalı sütunlar:")
            print('\n'.join(error_columns))
        else:
            print("Hatalı veri bulunmamaktadır.")
            self.print_separator()

    def analyze_columns(self):
        # Veri setinin ilk 5 satırını yazdırma
        print("Veri setindeki her sütunun ilk 5 satırı:")
        print(self.df.head())

        self.print_separator()

        print("Veri setindeki her sütunun son 5 satırı:")
        print(self.df.tail())

        self.print_separator()

        # Veri seti hakkında genel bilgileri yazdırma
        print(self.df.info())

        # Eksik değer kontrolü yapma
        error_columns = self.check_missing_values()

        # Her sütun için ayrı ayrı analiz işlemlerini gerçekleştirme
        for column in self.df.columns:
            self.print_separator()
            self.print_column_info(column)
            self.unique_values(column)

        # Veri analizinin tamamlandığına dair mesajı yazdırma
        self.print_separator()
        print("Veri analizi tamamlandı.")

        # Hatalı sütunları yazdırma
        self.print_error_columns(error_columns)


class PredictorOverCarData:
    def __init__(self, data_name_car):
        self.data_name_car = data_name_car
        self.df = None
        self.columns = None
        self.df_cleaned = None
        self.before_temp = None
        self.after_temp = None
        self.encode_temp = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        pd.set_option('display.max_columns', None)
        self.df = pd.read_csv(self.data_name_car)

    def preprocess_data(self):
        self.data_clean_and_correction()
        obj_col = self.df_cleaned.select_dtypes(include=['object']).columns
        self.data_transformation(obj_col)

    def data_clean_and_correction(self):
        self.df_cleaned = self.df.copy(deep=True)
        self.before_temp = self.df_cleaned.head()

        # Güncellenecek değerleri tanımla
        make_model_mapping = {
            'Nissan': 'Altima',
            'Ford': 'F-150',
            'Honda': 'Civic',
            'Toyota': 'Corolla',
            'Chevrolet': 'Silverado'
        }

        # 'Car Make' sütununu kullanarak 'Car Model' sütununu güncelle
        self.df_cleaned['Car Model'] = self.df_cleaned['Car Make'].map(
            make_model_mapping)

        self.after_temp = self.df_cleaned.head()

        return self.df_cleaned

    def data_transformation(self, categorical_cols):
        for col in categorical_cols:
            self.df_cleaned[col] = self.label_encoder.fit_transform(
                self.df_cleaned[col])

        self.encode_temp = self.df_cleaned.head()

    def visualize_model_predictions(self, X_test, y_test, y_pred, title, x_l, y_l):
        plt.scatter(X_test, y_test, color='blue')
        plt.plot(X_test, y_pred, color='red', linewidth=2)
        plt.xlabel(x_l)
        plt.ylabel(y_l)
        plt.title(title)
        plt.show()

    def predict_car_model_by_make(self):
        print("\nİlk veriler\n", self.before_temp[["Car Make", "Car Model"]])
        print("\nTemiz veriler\n", self.after_temp[["Car Make", "Car Model"]])
        print("\nDönüştürülmüş veriler\n",
              self.encode_temp[["Car Make", "Car Model"]])

        X = self.df_cleaned[["Car Make"]]
        y = self.df_cleaned["Car Model"]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # İlk 5 test verisinin x ve y verileri bastır
        first_5 = pd.concat([X_test.head(), y_test.head()], axis=1)
        first_5.columns = ['X_test_1', 'y_test_1']
        combined_df = pd.concat([first_5])
        print("\nİlk 5 test verisinin x ve y verileri\n", combined_df)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # tahminler yap
        y_pred = model.predict(X_test)
        print("\nTahminler\n", y_pred[:5])

        self.visualize_model_predictions(X_test, y_test, y_pred, "Araba Markası Model Tahmini", "Araba Markası",
                                         "Model")

    def predicte_next_model_price(self):
        print("\nİlk veriler\n", self.before_temp[["Car Model", "Car Year"]])
        print("\nTemiz veriler\n", self.after_temp[["Car Model", "Car Year"]])
        print("\nDönüştürülmüş veriler\n",
              self.encode_temp[["Car Model", "Car Year"]])

        # Veri setini özellikler ve hedef değişken olarak ayırma
        X = self.df_cleaned[["Car Model", "Car Year"]]
        y = self.df_cleaned["Sale Price"]

        # Eğitim ve test veri setlerini oluşturma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # İlk 5 test verisinin x ve y verileri bastır
        first_5 = pd.concat([X_test.head(), y_test.head()], axis=1)
        first_5.columns = ["X_test_1", "X_test_2", "y_test_1"]
        combined_df = first_5
        print("\nİlk 5 test verisinin x ve y verileri\n", combined_df)

        # Modeli eğitme
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Modelin performansını değerlendirme
        y_pred = model.predict(X_test)
        print("\nTahminler\n", y_pred[:5])

        mse = mean_squared_error(y_test, y_pred)
        print("\nOrtalama Karesel hata:", mse, "\n")

        # self.visualize_model_predictions(X_test, y_test, y_pred, "Gelecek Yıldaki Arac Modeli Satış Fiyat Tahminleri",
        # "Araba Modeli", "Satış Fiyatı")

    def predicte_future_car_price(self):
        print("\nİlk veriler\n", self.before_temp[[
              "Date", "Car Year", "Sale Price"]])
        print("\nTemiz veriler\n", self.after_temp[[
              "Car Model", "Car Year", "Sale Price"]])
        print("\nDönüştürülmüş veriler\n", self.encode_temp[[
              "Car Model", "Car Year", "Sale Price"]])

        X = self.df_cleaned[["Date"]]
        y = self.df_cleaned["Sale Price"]

        # Veri setini eğitim ve test olarak ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # İlk 5 test verisinin x ve y verileri bastır
        first_5 = pd.concat([X_test.head(), y_test.head()], axis=1)
        first_5.columns = ["X_test_1", "y_test_1"]
        combined_df = first_5
        print("\nİlk 5 test verisinin x ve y verileri\n", combined_df)

        # Lineer regresyon modelini oluştur ve eğit
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Test verileri üzerinde tahmin yap
        y_pred = model.predict(X_test)
        print("\nTahminler\n", y_pred[:5])

        # Hata metriklerini hesapla
        mse = mean_squared_error(y_test, y_pred)
        print("\nOrtalama Karesel hata:", mse, "\n")

        self.visualize_model_predictions(X_test, y_test, y_pred, "Gelecek Yıldaki Arac Satış Fiyat Tahminleri",
                                         "Araba Modeli", "Satış Fiyatı")

    def predict_successful_model_sales(self):
        print("\nİlk veriler\n", self.before_temp[[
              "Salesperson", "Car Model", "Commission Earned"]])
        print("\nTemiz veriler\n", self.after_temp[[
              "Salesperson", "Car Model", "Commission Earned"]])
        print("\nDönüştürülmüş veriler\n", self.encode_temp[[
              "Salesperson", "Car Model", "Commission Earned"]])

        # Veri setini özellikler ve hedef değişken olarak ayırma
        X = self.df_cleaned[["Salesperson", "Car Model"]]
        y = self.df_cleaned["Commission Earned"]

        # Eğitim ve test veri setlerini oluşturma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # İlk 5 test verisinin x ve y verileri bastır
        first_5 = pd.concat([X_test.head(), y_test.head()], axis=1)
        first_5.columns = ["X_test_1", "X_test_2", "y_test_1"]
        combined_df = first_5
        print("\nİlk 5 test verisinin x ve y verileri\n", combined_df)

        # Modeli eğitme
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Modelin performansını değerlendirme
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("\nOrtalama Karesel hata:", mse, "\n")

        # self.visualize_model_predictions(X_test, y_test, y_pred, "Satıcı Başarısı", "Araba Modeli", "Satış Başarısı")

    def estimate_total_sales_future_years(self):
        print("\nİlk veriler\n", self.before_temp[["Car Year", "Sale Price"]])
        print("\nTemiz veriler\n", self.after_temp[["Car Year", "Sale Price"]])
        print("\nDönüştürülmüş veriler\n",
              self.encode_temp[["Car Year", "Sale Price"]])

        # Veri setini özellikler ve hedef değişken olarak ayırma
        X = self.df_cleaned["Car Year"].values.reshape(-1, 1)
        y = self.df_cleaned["Sale Price"]

        # Eğitim ve test veri setlerini oluşturma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # İlk 5 test verisinin x ve y verileri bastır
        first_5 = pd.concat(
            [pd.DataFrame(X_test)[:5], pd.DataFrame(y_test)[:5]], axis=1)
        first_5.columns = ["X_test_1", "y_test_1"]
        combined_df = pd.concat([first_5])
        print("\nİlk 5 test verisinin x ve y verileri\n", combined_df)

        # Modeli eğitme
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Modelin performansını değerlendirme
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("\nOrtalama Karesel hata:", mse, "\n")

        # self.visualize_model_predictions(X_test, y_test, y_pred, "Hangi satıcı gelecek yıllarda ne kadar satış yapacak",
        #                                  "Satıcı", "Yapılan Satış")

    def predict_sales_success_based_on_production_date(self):
        print("\nİlk veriler\n", self.before_temp[[
              "Car Year", "Commission Earned"]])
        print("\nTemiz veriler\n", self.after_temp[[
              "Car Year", "Commission Earned"]])
        print("\nDönüştürülmüş veriler\n",
              self.encode_temp[["Car Year", "Commission Earned"]])

        X = self.df_cleaned["Car Year"]
        y = self.df_cleaned["Commission Earned"]

        # Eğitim ve test veri setlerini oluşturma
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # İlk 5 test verisinin x ve y verileri bastır
        first_5 = pd.concat([X_test.head(), y_test.head()], axis=1)
        first_5.columns = ["X_test_1", "y_test_1"]
        combined_df = first_5
        print("\nİlk 5 test verisinin x ve y verileri\n", combined_df)

        # Modeli eğitme
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Modelin performansını değerlendirme
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("\nOrtalama Karesel hata:", mse, "\n")

        self.visualize_model_predictions(X_test, y_test, y_pred, "Araba üretim tarihine göre satış başarısı",
                                         "Araba Yılı", "Satış Başarısı")


def baslat():
    predictor = PredictorOverCarData("car_sales_data.csv")
    predictor.load_data()

    df = predictor.df

    column_stats = DataAnalyzer(df)
    column_stats.analyze_columns()

    predictor.preprocess_data()

    # predictor.predict_car_model_by_make()

    # predictor.predicte_next_model_price()

    # predictor.predicte_future_car_price()

    # predictor.predict_successful_model_sales()

    # predictor.estimate_total_sales_future_years()

    # predictor.predict_sales_success_based_on_production_date()


baslat()
