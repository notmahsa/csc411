import numpy as np
import scipy as sc
import sklearn
from sklearn.feature_extraction import text
from sklearn.tree import DecisionTreeClassifier as dtc

with open('clean_real.txt') as f:
    real = f.read().splitlines()

with open('clean_fake.txt') as f:
    fake = f.read().splitlines()

data = np.array([[i, 1] for i in real] + [[i, 0] for i in fake[:]])
np.random.shuffle(data)

N = len(data)

count_vectorizer = text.CountVectorizer()

train_d = [i[0] for i in data[:int(N*0.7)]]
vdxn_d = [i[0] for i in data[int(N*0.7):int(N*0.85)]]
test_d = [i[0] for i in data[int(N*0.85):]]

train_counts = count_vectorizer.fit_transform(train_d)
vdxn_counts = count_vectorizer.transform(vdxn_d)
test_counts = count_vectorizer.transform(test_d)

train_t = [i[1] for i in data[:int(N*0.7)]]
vdxn_t = [i[1] for i in data[int(N*0.7):int(N*0.85)]]
test_t = [i[1] for i in data[int(N*0.85):]]

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

clf = dtc(max_depth=3)
clf.fit(X=train_counts, y=train_t)
clf.score(X=vdxn_counts, y=vdxn_t)


clf = dtc(max_depth=1)
clf.fit(X=train_counts, y=train_t)
clf.score(X=vdxn_counts, y=vdxn_t)
