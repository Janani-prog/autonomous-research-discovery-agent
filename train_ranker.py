from sklearn.linear_model import LinearRegression
import joblib

from features import RANKER_FEATURE_KEYS


def train_ranker(X, y, y_range):
    """
    Fits a LinearRegression and saves it as a bundle (model + the feature
    keys it was trained on + the label normalization range) rather than the
    bare model. scoring.py checks feature_keys match before trusting a
    loaded ranker.pkl, so a future feature-set change can't silently get
    served with mismatched-meaning inputs the way the old feature-order
    inconsistency did.
    """
    model = LinearRegression()
    model.fit(X, y)

    bundle = {
        "model": model,
        "feature_keys": RANKER_FEATURE_KEYS,
        "y_min": y_range[0],
        "y_max": y_range[1],
    }
    joblib.dump(bundle, "ranker.pkl")
    return bundle
