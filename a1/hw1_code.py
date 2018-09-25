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

count_vectorizer = text.CountVectorizer(stop_words='english')

train_d = count_vectorizer.fit([i[0] for i in data[:int(N*0.7)]])
vdxn_d = count_vectorizer.fit([i[0] for i in data[int(N*0.7):int(N*0.85)]])
test_d = count_vectorizer.fit([i[0] for i in data[int(N*0.85):]])
train_t = text.CountVectorizer([i[1] for i in data[:int(N*0.7)]])
vdxn_t = text.CountVectorizer([i[1] for i in data[int(N*0.7):int(N*0.85)]])
test_t = text.CountVectorizer([i[1] for i in data[int(N*0.85):]])

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

clf = dtc(max_depth=3)
