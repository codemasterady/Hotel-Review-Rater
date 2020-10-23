# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
# Downloading stopwords
nltk.download('stopwords')


class NeuralEngine:
    def __init__(self):
        # Getting the DataSet
        self.nb = GaussianNB()
        pass

    #! Training & Creating the Classifier model
    def classify_train(self):
        dataset = pd.read_csv(
            r"C:\Users\Selvaseetha\Repository\Ratings Predictor\models\tripadvisor_hotel_reviews.csv").values
        # Cleaning The Text
        corpus = []
        for i in range(0, 10000):
            review = re.sub("^a-zA-Z", ' ', dataset[i, 0])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')  # Ignores the not value
            review = [ps.stem(word)
                      for word in review if not word in set(all_stopwords)]
            review = " ".join(review)
            corpus.append(review)

        # Implementing the Bag Of Words Model (Count Vectorizer)

        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=47000)
        X = cv.fit_transform(corpus).toarray()

        # Getting the dependant variable

        y = dataset[0:10000, 1]

        # Splitting the data into the training & test set

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1)

        # Using Naive Bayes to train
        y = y.astype('int')  # ! To solidify the type
        y = y.reshape(-1, 1)
        self.nb.fit(X, np.array(y))
        pass

    #! Predicting the final Outputs
    def predict_classifier(self, text):
        review = re.sub("^a-zA-Z", ' ', text)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')  # Ignores the not value
        review = [ps.stem(word)
                  for word in review if not word in set(all_stopwords)]
        review = " ".join(review)
        review = np.array(review).reshape(-1, 1)
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=47000)
        X = cv.fit_transform(review).toarray()
        X = np.array(X).reshape(-1, 1)
        y_pred = self.nb.predict(X)
        print(y_pred)
        pass
