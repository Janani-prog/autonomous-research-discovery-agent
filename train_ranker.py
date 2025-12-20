from sklearn.linear_model import LinearRegression
import joblib

def train_ranker(X, y):
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "ranker.pkl")
