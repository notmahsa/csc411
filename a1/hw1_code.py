import numpy as np
import scipy as sc
import sklearn
from sklearn.feature_extraction import text
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import export_graphviz as eg

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

max_depths = [1, 10, 100, 1000, 10000]
criteria = ["gini", "entropy"]

accuracy = []
trees = []
for d in max_depths:
    for c in criteria:
        clf = dtc(criterion=c, max_depth=d, random_state=0)
        oppress_output = clf.fit(X=train_counts, y=train_t)
        # To validate accuracy calculated: accuracy.append(clf.score(X=vdxn_counts, y=vdxn_t))
        pred = clf.predict(X=vdxn_counts)
        accuracy.append(1 - np.mean(pred!=vdxn_t))
        trees.append(clf)

M = np.argmax(accuracy)
eg(trees[M], out_file='tree.dot', max_depth=2)
print M
