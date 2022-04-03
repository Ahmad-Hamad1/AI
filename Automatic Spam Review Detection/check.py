import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords  # Import stop words module
from nltk.stem.porter import PorterStemmer  # Import PorterStemmer which will be used to make all verbs in present.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow as tf

dataset = pd.read_csv("Output.csv", quoting=3, low_memory=False)  # quoting = 3 tells the function to ignore quotations.

#print(dataset.isnull().sum())

inputData = dataset.iloc[:50000, [False, False, True, True, True, True, False, True, True, True]].values
imp_most = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp_most.fit(inputData[:, [3, 6]])
inputData[:, [ 3, 6]] = imp_most.transform(inputData[:, [ 3, 6]])

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(inputData[:, [1, 2, 4, 5]])
inputData[:, [1, 2, 4, 5]] = imp_mean.transform(inputData[:, [1, 2, 4, 5]])

# print(inputData[22026, 3])
# exit(1)

nltk.download("stopwords")  # Downloading stop words module.
cleanedReviews = []
days = []
months = []
years = []
updated = []
number_of_rows = np.shape(inputData)[0]
spam=0
not_spam=0
for i in range(0, number_of_rows):
    #print(i)
    # print(inputData[i][0])
    # print(i)
    rev = inputData[i][0]
    review = re.sub("[^A-Za-z]", ' ', rev)  # Remove all special characters and numbers.
    review = review.lower()  # Convert all letters to lower case.
    review = review.split()  # Split the words in the review.
    ps = PorterStemmer()  # Creating a stemmer object.
    stops = stopwords.words("english")  # Get all english stop words
    stops.remove("not")  # Remove not from the stop words list.
    review = [ps.stem(word) for word in review if word not in set(stops)]
    review = " ".join(review)  # Join all words together after stemming them and removing stopwords.
    cleanedReviews.append(review)
    if(inputData[i][6]==1):
        spam+=1
    else:
        not_spam+=1
    date = inputData[i][3]
    date = re.split("/|- ", date)
    index = 0
    if str(date[index]).lower().strip() == "update":
        updated.append(1.0)
        index += 1
    else:
        updated.append(0.0)
    months.append(float(date[index]))
    index += 1
    days.append(float(date[index]))
    index += 1
    years.append(float(date[index]))

print("--------------------")
print(spam)

print(not_spam)


# Creating the bag of words model.

cv = CountVectorizer(max_features=40000)
data = cv.fit_transform(cleanedReviews).toarray()  # Add words of the cleaned reviews to the bag of words.
columns = len(inputData[0]) + len(data[0]) + 1
features = np.arange(np.shape(inputData)[0] * columns).reshape((np.shape(inputData)[0], columns))
features = features.astype("float32")
features[:, 0:len(data[0])] = data
features[:, len(data[0]):-4] = inputData[:, [1, 2, 4, 5]]
features[:, -4] = updated
features[:, -3] = months
features[:, -2] = days
features[:, -1] = years
filtered = inputData[:, -1]  # Get the output of the CSV file.
filtered = filtered.astype("int32")

# print(features)


# Naive Bayes Model.
features_train, features_test, filtered_train, filtered_test = train_test_split(features, filtered, test_size=0.20, random_state=50)
#
# classifier = GaussianNB()
# classifier.fit(features_train, filtered_train)
# filtered_predicted = classifier.predict(features_test)
# print(np.concatenate((filtered_predicted.reshape(len(filtered_predicted), 1), filtered_test.reshape(len(filtered_test), 1)), 1))
# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='sigmoid'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(features_train, filtered_train, batch_size = 32, epochs = 195)
filtered_predicted = ann.predict(features_test)
filtered_predicted = (filtered_predicted > 0.5)
#print(np.concatenate((filtered_predicted.reshape(len(filtered_predicted), 1), filtered_test.reshape(len(filtered_test), 1)), 1))
# Making the confusion matrix.
cm = confusion_matrix(filtered_test, filtered_predicted)
print("The Confusion Matrix: ")
print(cm)

tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
pr = tp / (tp + fp)
rec = tp / (tp + fn)

print("The Precision Is : ", pr)
print("The Recall Is : ", rec)
print("The F1 - Score Is : ", (2*pr*rec)/(pr+rec))
print("Accuracy Is : ", accuracy_score(filtered_test, filtered_predicted))
