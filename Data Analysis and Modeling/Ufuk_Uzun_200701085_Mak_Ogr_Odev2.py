import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb


class NBAAnalyze:
    def __init__(self, data_name_nba):
        self.data_name_nba = data_name_nba
        self.df = None
        self.column_stats = None
        self.df_cleaned = None

    def nba_load_data(self):
        self.df = pd.read_csv(self.data_name_nba)

    def nba_analyze_columns(self):
        self.column_stats = self.df.describe()

    def nba_clean_data(self):
        numerical_columns = [
            'HOME_FG', 'AWAY_FG', 'HOME_FGA', 'AWAY_FGA', 'HOME_FG_PCT', 'AWAY_FG_PCT',
            'HOME_FG3', 'AWAY_FG3', 'HOME_FG3A', 'AWAY_FG3A', 'HOME_FG3_PCT', 'AWAY_FG3_PCT',
            'HOME_FT', 'AWAY_FT', 'HOME_FTA', 'AWAY_FTA', 'HOME_FT_PCT', 'AWAY_FT_PCT',
            'HOME_OFF_REB', 'AWAY_OFF_REB', 'HOME_DEF_REB', 'AWAY_DEF_REB',
            'HOME_TOT_REB', 'AWAY_TOT_REB', 'HOME_AST', 'AWAY_AST', 'HOME_STL', 'AWAY_STL',
            'HOME_TURNOVERS', 'AWAY_TURNOVERS', 'HOME_BLK', 'AWAY_BLK', 'HOME_PTS', 'AWAY_PTS'
        ]
        self.df_cleaned = self.df[numerical_columns].copy()
        self.df_cleaned.fillna(self.df_cleaned.mean(), inplace=True)

    def nba_cluster_teams_kmeans(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df_cleaned)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(scaled_data)
        self.team_labels_kmeans = kmeans.labels_
        clusterlabels = kmeans.labels_
        self.df['Cluster'] = clusterlabels
        print("\nTeam Labels (K-Means):")
        print(self.df[['HOME', 'AWAY', 'Cluster']])

    def nba_cluster_teams_gmm(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df_cleaned)
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(scaled_data)
        cluster_labels = gmm.predict(scaled_data)
        self.df['Cluster'] = cluster_labels
        print("\nTeam Labels (GMM):")
        print(self.df[['HOME', 'AWAY', 'Cluster']])

    def nba_print_results(self):
        print("Column Statistics:")
        print(self.column_stats)


class MCDonaldsAnalyze:
    def __init__(self, data_name_mcd):
        self.data_name_mcd = data_name_mcd
        self.df = None
        self.column_stats = None
        self.df_cleaned = None

    def load_data(self):
        self.df = pd.read_csv(self.data_name_mcd)

    def mcdon_analyze_columns(self):
        self.column_stats = self.df.describe()

    def clean_data_mcdonalds(self):
        self.df['Per Serve Size'] = self.df['Per Serve Size'].str.replace('[^\d.]', '', regex=True).astype(float)

    def xgb_energy_mcdonalds(self):
        X = self.df[['Per Serve Size', 'Protein (g)', 'Total fat (g)', 'Sat Fat (g)', 'Trans fat (g)',
                     'Cholesterols (mg)', 'Total carbohydrate (g)', 'Total Sugars (g)', 'Added Sugars (g)', 'Sodium (mg)']]
        y = self.df['Energy (kCal)']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print("XGB Performans Metrigi:", rmse)

    def mcdon_print_results(self):
        print("Column Statistics:")
        print(self.column_stats)


data_name_nba = 'game_scores.csv'
"""
nba_analyzer = NBAAnalyze(data_name_nba)
nba_analyzer.nba_load_data()
nba_analyzer.nba_analyze_columns()
nba_analyzer.nba_clean_data()
nba_analyzer.nba_cluster_teams_kmeans()
nba_analyzer.nba_cluster_teams_gmm()
nba_analyzer.nba_print_results()"""


data_name_mcd = 'india_menu.csv'

mcdon_analyzer = MCDonaldsAnalyze(data_name_mcd)
mcdon_analyzer.load_data()
mcdon_analyzer.mcdon_analyze_columns()
mcdon_analyzer.clean_data_mcdonalds()
mcdon_analyzer.xgb_energy_mcdonalds()
mcdon_analyzer.mcdon_print_results()
