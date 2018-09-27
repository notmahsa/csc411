import numpy as np
import scipy as sc
import sklearn
from sklearn.feature_extraction import text
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import export_graphviz as eg


def load_data():
    # load headlines
    with open('clean_real.txt') as f:
        real = f.read().splitlines()

    with open('clean_fake.txt') as f:
        fake = f.read().splitlines()

    # flag headlines with Real=1 and Fake=0 and shuffle them in one array
    data = np.array([[i, 1] for i in real] + [[i, 0] for i in fake[:]])
    np.random.shuffle(data)
    N = len(data)
    count_vectorizer = text.CountVectorizer()

    # separate training, validation, and test data
    train_d = [i[0] for i in data[:int(N*0.7)]]
    vdxn_d = [i[0] for i in data[int(N*0.7):int(N*0.85)]]
    test_d = [i[0] for i in data[int(N*0.85):]]

    # vectorize data
    train_counts = count_vectorizer.fit_transform(train_d)
    vdxn_counts = count_vectorizer.transform(vdxn_d)
    test_counts = count_vectorizer.transform(test_d)

    # separate training, validation and test targets
    train_t = [i[1] for i in data[:int(N*0.7)]]
    vdxn_t = [i[1] for i in data[int(N*0.7):int(N*0.85)]]
    test_t = [i[1] for i in data[int(N*0.85):]]

    return train_counts, vdxn_counts, test_counts, \
           train_t, vdxn_t, test_t, count_vectorizer.get_feature_names()


def select_model(train_counts, vdxn_counts, test_counts,
                 train_t, vdxn_t, test_t, features):
    max_depths = [1, 10, 50, 100, 200]
    criteria = ["gini", "entropy"]
    accuracy = []
    trees = []

    # iterate through max_depth values and critera to calculate accuracy
    for d in max_depths:
        for c in criteria:
            clf = dtc(criterion=c, max_depth=d, random_state=0)
            oppress_output = clf.fit(X=train_counts, y=train_t)
            pred = clf.predict(X=vdxn_counts)
            # calculate accuracy as "1 - error"
            accuracy.append(1 - np.mean(pred!=vdxn_t))
            print "Criterion:", c, "& Max Depth:", d, \
                  "| Accuracy:", 1 - np.mean(pred!=vdxn_t)
            trees.append(clf)

    # find tree with maximum accuracy
    M = np.argmax(accuracy)

    # visualize tree as a .dot file
    eg(trees[M], out_file='tree.dot', max_depth=2,filled=True, rounded=True, \
       feature_names=features, class_names=["Fake","Real"])
    return trees[M]


def information_gain(term, tree):
    # fing split with given term eg "trump"
    ig = 0.
    for i in range(len(tree.tree_.feature)):
        if features[tree.tree_.feature[i]] == term:
            # find entropy at split node
            sum = tree.tree_.value[i][0][0] + tree.tree_.value[i][0][1]
            if sum == 0:
                break
            p1 = tree.tree_.value[i][0][0] / float(sum)
            h1 = - p1 * np.log2(p1) - (1 - p1) * np.log2((1 - p1))
            #find child nodes
            left = tree.tree_.children_left[i]
            right = tree.tree_.children_right[i]
            # find entropy at right node
            sum_right = tree.tree_.value[right][0][0] + tree.tree_.value[right][0][1]
            p2 = tree.tree_.value[right][0][0] / float(sum_right)
            if sum_right != 0 and 0 < p2 < 1:
                h2 = - p2 * np.log2(p2) - (1 - p2) * np.log2((1 - p2))
            else:
                h2 = 0
            # find entropy at left node
            sum_left = tree.tree_.value[left][0][0] + tree.tree_.value[left][0][1]
            p3 = tree.tree_.value[left][0][0] / float(sum_left)
            if sum_left != 0 and 0 < p3 < 1:
                h3 = - p3 * np.log2(p3) - (1 - p3) * np.log2((1 - p3))
            else:
                h3 = 0
            # return information gain
            ig += h1 - (sum_right / float(sum)) * h2 - (sum_left / float(sum)) * h3

    print term, ig


if __name__ == "__main__":
    tc, vc, sc, tt, vt, st, features = load_data()
    tree = select_model(tc, vc, sc, tt, vt, st, features)
    terms = ["trump", "win", "war", "america", "tribute"]
    for i in terms:
        information_gain(i, tree)
