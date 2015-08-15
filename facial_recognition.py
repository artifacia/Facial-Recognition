# import the necessary packages
# the basic packages
from time import time
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

# importing the machine learning packages
from sklearn.datasets import fetch_lfw_people
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Downloading the data(labeled faces in the wild) from the servers and if the data is already present,
# in the current working directory then load then as a numpy array
lfw_people = fetch_lfw_people(min_faces_per_person=70)

# the basic height and the width of the images and the number of images in the dataset
n_samples, h, w = lfw_people.images.shape

# load the data into a variable and its target values or say expected values in another variable
X = lfw_people.data
y = lfw_people.target

n_images = X.shape[0]
n_features = X.shape[1]
person_names = lfw_people.target_names
n_classes = person_names.shape[0]
print person_names

# printing the information about the dataset
print "Total dataset size:"
print "# images: %d" % n_images
print "# features: %d" % n_features
print "# classes: %d" % n_classes

# split into a training and testing set by 70-30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Train a Logisticregression classification model
print "Fitting the classifier to the training set"
t0 = time()

logisticreg = LogisticRegression()
logisticreg = logisticreg.fit(X_train, y_train)
print "done in %0.3fs" % (time() - t0)

with open("model.pkl", "wb") as fh:
    pickle.dump(logisticreg, fh)

# Quantitative evaluation of the model quality on the test set
print "Predicting people's names on the test set"
t0 = time()
y_pred = logisticreg.predict(X_test)
print "done in %0.3fs" % (time() - t0)

# calculating the accuracy of the current system
num_examples = y_pred.shape[0]
count = 0 
for idx in xrange(num_examples):
    if y_pred[idx] == y_test[idx]:
        count += 1

print "Accuracy: %0.3fs" % (count*100.0/num_examples)

print confusion_matrix(y_test, y_pred, labels=range(n_classes))


# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in xrange(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, person_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

plt.show()