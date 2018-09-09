from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


classifiers = [
    ("Nearest Neighbors", KNeighborsClassifier(3)),
    ("Linear SVM", SVC(kernel="linear", C=0.025)),
    ("RBF SVM", SVC(gamma=2, C=1)),
    # ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    ("Random Forest", RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1)),
    ("Neural Net", MLPClassifier(hidden_layer_sizes=(10,))),
    ("AdaBoost", AdaBoostClassifier()),
    # ("Naive Bayes", GaussianNB()),
    # ("QDA", QuadraticDiscriminantAnalysis()),
]


# Preprocessing
print("\nPREPROCESSING")

# Read preprocessed training data

csv_train = read_csv("preprocessed-training.csv")
csv_test = read_csv("preprocessed-test.csv")

print("Preprocessed data has been read")

# Input training lemmas to vectorizer

vectorizer = TfidfVectorizer()
vector_train = vectorizer.fit_transform(csv_train.lemmas)

print("Training lemmas has been fit and transformed")

# Input test lemmas to vectorizer

vector_test = vectorizer.transform(csv_test.lemmas)

print("Testing lemmas has been transformed")


# Learning
for classifier_name, classifier in classifiers:
    print("\nLEARNING: " + classifier_name)

    # Fit training lemmas
    classifier.fit(vector_train, csv_train.predictions)
    print("Classifier model has been trained")

    # Predict testing lemmas
    prediction = classifier.predict(vector_test)

    n_same = 0
    for i, val in enumerate(csv_test.predictions):
        if val == prediction[i]:
            n_same += 1

    p_accuracy = n_same / len(csv_test.predictions)
    print("Accuracy: {}%".format(p_accuracy * 100))
