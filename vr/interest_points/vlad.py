###################################################### Environment variables #################################
RANDOM_STATE = 42
ITERATIONS = 3
SAMPLES = 5000
DESCRIPTOR_DIM = 64
##############################################################################################################

import time
from contextlib import contextmanager

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# !pip install opencv-contrib-python==3.4.2.17
# !pip install -U scikit-learn

(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
# We will do our preprocessing so taking stacking all of them together
xtrain = np.vstack([xtrain, xtest])
ytrain = np.vstack([ytrain, ytest])

# select random samples
np.random.seed(RANDOM_STATE)
randomSampleIndices = np.random.choice(xtrain.shape[0], SAMPLES, replace=False)
xtrain = xtrain[randomSampleIndices, :, :, :]
ytrain = ytrain[randomSampleIndices]

print('xtrain: ', xtrain.shape)
print('ytrain: ', ytrain.shape)
print('Occurances of classes', np.unique(ytrain, return_counts=True), sep='\n')


@contextmanager
def timer(task_name="Timer"):
    print("{} started...".format(task_name))
    t0 = time.time()
    yield
    print("{} done in {:.0f} seconds...\n".format(task_name, time.time() - t0))


def preprocess(x, y):
    labels = []
    descriptors = []
    index = 0
    sift = cv.xfeatures2d.SURF_create()
    descStacked = None

    for img in x:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # img = cv.equalizeHist(img)
        img = cv.resize(img, (50, 50), interpolation=cv.INTER_AREA)
        _, d = sift.detectAndCompute(img, None)

        if d is not None:
            descriptors.append(d)
            labels.append(int(y[index]))
            if descStacked is None:
                descStacked = d
            else:
                descStacked = np.vstack((descStacked, d))

        index += 1
        if index % 1000 == 0:
            print('Processed', index, 'images...')

    return descriptors, labels, descStacked


with timer('Getting SIFT descriptors'):
    descriptors, labels, descStacked = preprocess(xtrain, ytrain)

from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score


def train(K):
    # Perform k means
    with timer('\t kmeans'):
        codebook, distance = kmeans(descStacked, K, ITERATIONS)

    # contruct bag of words representation
    with timer('\t constructing BoVW Vlad'):
        n = len(labels)
        features = np.zeros((n, K, DESCRIPTOR_DIM), 'float32')
        for i in range(n):
            # get cluster assignment using codebook
            clusters, distortions = vq(descriptors[i], codebook)
            # for each cluster accumulate the distances from cluster centre
            for j in range(clusters.shape[0]):
                clusterNo = clusters[j]
                clusterVector = codebook[clusterNo]
                descriptorVector = descriptors[i][j]
                residualVector = clusterVector - descriptorVector
                features[i][clusterNo] += residualVector

            for j in range(clusters.shape[0]):
                clusterNo = clusters[j]
                vec = features[i][clusterNo]
                vec = np.sign(vec) * np.sqrt(abs(vec))
                features[i][clusterNo] = vec / np.sqrt(np.dot(vec, vec))

    with timer('\t reshaping to kx64'):
        features = features.reshape((n, K * DESCRIPTOR_DIM))

    # Remove mean ans scale to unit variance
    with timer('\t standardizing'):
        features = StandardScaler().fit(features).transform(features)

    with timer('\t test train splitting'):
        xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.2, random_state=RANDOM_STATE)

    print(xtrain.shape)

    with timer('\t training'):
        model = LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=500)
        # model = RandomForestClassifier()
        # model = lgb.LGBMClassifier()
        model.fit(xtrain, ytrain)
        predictions = model.predict_proba(xtest)

    return accuracy_score(ytest, predictions)


allPredictions = {}
for k in range(10, 201, 20):
    with timer('Training for k = {}'.format(k)):
        allPredictions[k] = train(k)

print(allPredictions)

plt.xlabel('K')
plt.ylabel('Accuracy')
plt.plot(*zip(*allPredictions.items()))
plt.show()
