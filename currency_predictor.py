import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
import matplotlib.pyplot as plt

class CurrencyPredictor:
    def __init__(self, data_file, model_file=None, random_state=42):
        self.data_file = data_file
        self.model_file = model_file
        self.random_state = random_state

    def load_data(self):
        return pd.read_csv(self.data_file)

    def preprocess_data(self, data):
        data['data'] = pd.to_datetime(data['data'])
        data['year'] = data['data'].dt.year
        data['month'] = data['data'].dt.month
        data['day'] = data['data'].dt.day
        X = data[['year', 'month', 'day']]
        y = data['curs']
        return X, y

    def create_model(self):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(random_state=self.random_state))
        ])
        return model

    def train_model(self, model, X, y):
        model.fit(X, y)
        if self.model_file:
            dump(model, self.model_file)
        return model

    def evaluate_model(self, model, X, y, scoring=None):
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        scores = cross_val_score(model, X, y, scoring=scoring, cv=5)
        return scores

    def predict(self, model, X):
        return model.predict(X)

    def plot_results(self, y_true, y_pred):
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='True')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        plt.show()

    def run(self):
        data = self.load_data()
        X, y = self.preprocess_data(data)
        model = self.create_model()
        if self.model_file and self.model_file.exists():
            model = load(self.model_file)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
            model = self.train_model(model, X_train, y_train)
            y_pred = self.predict(model, X_test)
            print("Test MSE:", mean_squared_error(y_test, y_pred))
            print("Test MAE:", mean_absolute_error(y_test, y_pred))
            print("Test R^2:", r2_score(y_test, y_pred))
            self.plot_results(y_test, y_pred)


if __name__ == "__main__":
    predictor = CurrencyPredictor('curs_dol.csv')
    predictor.run()
