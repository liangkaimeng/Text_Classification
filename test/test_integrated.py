#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : lkm

import warnings
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

""" 导入数据 """
# 导入数据
categories = ['alt.atheism', 'rec.sport.hockey', 'comp.graphics', 'sci.crypt', 'comp.os.ms-windows.misc',
              'sci.electronics', 'comp.sys.ibm.pc.hardware', 'sci.med', 'comp.sys.mac.hardware', 'sci.space',
              'comp.windows.x', 'soc.religion.christian', 'misc.forsale', 'talk.politics.guns', 'rec.autos',
              'talk.politics.mideast', 'rec.motorcycles', 'talk.politics.misc', 'rec.sport.baseball', 'talk.religion.misc']
# 导入训练数据
train_path = r'D:\LKM\MachineLearning\TextClassification\data\20news-bydate-train'
train_data = load_files(container_path=train_path, categories=categories)
# 导入评估数据
test_path = r'D:\LKM\MachineLearning\TextClassification\data\20news-bydate-test'
test_data = load_files(container_path=test_path, categories=categories)

""" 数据准备与理解 """
# 计算词频
count_vect = CountVectorizer(stop_words='english', decode_error='ignore')
train_count = count_vect.fit_transform(train_data.data)
# 计算TF-IDF
tf_transformer = TfidfVectorizer(stop_words='english', decode_error='ignore')
x = tf_transformer.fit_transform(train_data.data)
y = train_data.target

""" 设置评估算法的基准 """
num_folds = 10
seed = 7
scoring = 'accuracy'

""" 集成算法 """
ensembles = {}
ensembles['RF'] = RandomForestClassifier()
ensembles['AB'] = AdaBoostClassifier()
# 比较集成算法
results = []
for key in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_result = cross_val_score(ensembles[key], x, y, cv=kfold, scoring=scoring)
    results.append(cv_result)
    print('%s : %f (%f)' % (key, cv_result.mean(), cv_result.std()))
# 箱线图比较算法
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()

