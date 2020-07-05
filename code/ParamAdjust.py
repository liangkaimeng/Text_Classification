#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import warnings
from joblib import dump
from sklearn.datasets import load_files
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

file = r"D:\LKM\MachineLearning\TextClassification\data\20news-bydate-train"
categories = ['alt.atheism', 'rec.sport.hockey', 'comp.graphics', 'sci.crypt', 'comp.os.ms-windows.misc',
              'sci.electronics', 'comp.sys.ibm.pc.hardware', 'sci.med', 'comp.sys.mac.hardware', 'sci.space',
              'comp.windows.x', 'soc.religion.christian', 'misc.forsale', 'talk.politics.guns', 'rec.autos',
              'talk.politics.mideast', 'rec.motorcycles', 'talk.politics.misc', 'rec.sport.baseball', 'talk.religion.misc']

class Logistic:
    def __init__(self):
        train = load_files(container_path=file, categories=categories)
        tf_transformer = TfidfVectorizer(stop_words='english', decode_error='ignore')
        self.x = tf_transformer.fit_transform(train.data)
        self.y = train.target

        self.paramGrid()

    def paramGrid(self):
        params = {}
        params['C'] = [0.1, 5, 13, 15]
        model = LogisticRegression()
        kfold = KFold(n_splits=10, random_state=7)
        grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=kfold)
        grid_result = grid.fit(X=self.x, y=self.y)
        C = grid_result.best_params_
        model = LogisticRegression(C=C['C'])
        model.fit(self.x, self.y)
        dump(model, 'LogistModel.joblib')

if __name__ == '__main__':
    Logistic()