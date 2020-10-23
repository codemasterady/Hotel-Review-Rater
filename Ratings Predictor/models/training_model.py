from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
# nltk.download('stopwords')
dataset = pd.read_csv("tripadvisor_hotel_reviews.csv")

class trainer:
    def __init__(self, dataset):
        self.dataset = dataset
        pass
    
    #! Converts text dataset into Bag Of Words
    def text_to_array(self, max_lim =47000):
        corpus = []
        for i in range(0, 20000):
                    review = re.sub("^a-zA-Z", ' ', self.dataset["Review"][i])
                    review = review.lower()
                    review = review.split()
                    ps = PorterStemmer()
                    all_stopwords = stopwords.words('english')
                    all_stopwords.remove('not')
                    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
                    review = " ".join(review)
                    corpus.append(review)  
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features= max_lim)
        X = cv.fit_transform(corpus).toarray()
        print(len(X[0]))
        return np.array(X).reshape(-1, 1)
    
    #! Uses the Naive-Bayes Classifier
    def classifier(self):
        pass
    
    #! Trains a neural network
    def train_model(self):
        pass
        
        

      
