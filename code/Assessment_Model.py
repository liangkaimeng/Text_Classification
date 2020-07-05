#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

from joblib import load
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

file = r"D:\LKM\MachineLearning\TextClassification\data\20news-bydate-test"
categories = ['alt.atheism', 'rec.sport.hockey', 'comp.graphics', 'sci.crypt', 'comp.os.ms-windows.misc',
              'sci.electronics', 'comp.sys.ibm.pc.hardware', 'sci.med', 'comp.sys.mac.hardware', 'sci.space',
              'comp.windows.x', 'soc.religion.christian', 'misc.forsale', 'talk.politics.guns', 'rec.autos',
              'talk.politics.mideast', 'rec.motorcycles', 'talk.politics.misc', 'rec.sport.baseball', 'talk.religion.misc']
model_file = r"D:\LKM\MachineLearning\TextClassification\code\LogistModel.joblib"

def Model_Assess():
    dataset_test = load_files(container_path=file, categories=categories)
    tf_transformer = TfidfVectorizer(stop_words='english', decode_error='ignore')
    x = tf_transformer.fit_transform(dataset_test.data)
    y = dataset_test.target
    model = load(model_file)
    y0 = model.predict(x)
    print('准确度：', accuracy_score(y, y0))
    print('验证结果：', classification_report(y, y0))

if __name__ == '__main__':
    Model_Assess()