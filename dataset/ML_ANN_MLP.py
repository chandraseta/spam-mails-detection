from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier


# Training
print ("\nTRAINING")
# Read preprocessed training data

csv_train = read_csv("preprocessed-training.csv")

print("Preprocessed data has been read")

# Input to vectorizer

vectorizer = TfidfVectorizer()
vector_train = vectorizer.fit_transform(csv_train.lemmas)

print("Lemmas has been feed to vectorizer")

# Input to classifier

classifier = MLPClassifier(
    hidden_layer_sizes=(5, 2), random_state=1)
classifier.fit(vector_train, csv_train.predictions)

print("Classifier model has been fit")


# Testing
print ("\nTESTING")
# Read preprocessed test data

csv_test = read_csv("preprocessed-test.csv")

print("Preprocessed data has been read")

# Input to vectorizer

vector_test = vectorizer.transform(csv_test.lemmas)

print("Lemmas has been feed to vectorizer")

# Input to classifier

prediction = classifier.predict(vector_test)

print("Lemmas has been predicted")

# Performance measure

n_same = 0
for i, val in enumerate(csv_test.predictions):
    if val == prediction[i]:
        n_same += 1

p_accuracy = n_same / len(csv_test.predictions)
print("Accuracy: {}%".format(p_accuracy * 100))
