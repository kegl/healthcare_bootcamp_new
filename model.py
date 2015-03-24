from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = Pipeline([('imputer', Imputer()), 
            ('rf', RandomForestClassifier(n_estimators=100))])
    
    def __getattr__(self, attrname):
        return getattr(self.clf, attrname)
