from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.svm import NuSVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.base import BaseEstimator

lass Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([('imputer', Imputer()), 
            ('rf', RandomForestClassifier(n_estimators=300))])
    
        self.clf2 = Pipeline([('imputer', Imputer(strategy='most_frequent')),
        ('svc', NuSVC(probability=True))])

    def fit(self, X, y):
        self.clf.fit(X, y)
        X_good_features = self.clf.transform(X)
        X_scaled = preprocessing.scale(X_good_features)
        self.clf2.fit(X_scaled, y)

    def predict(self, X):
        X_good_features = self.clf.transform(X)
        X_scaled = preprocessing.scale(X_good_features)
        return self.clf2.predict(X_scaled)

    def predict_proba(self, X):
        X_good_features = self.clf.transform(X)
        X_scaled = preprocessing.scale(X_good_features)
        return self.clf2.predict_proba(X_scaled)

