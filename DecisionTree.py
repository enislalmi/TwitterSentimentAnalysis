import random
import pandas as pd
import numpy as np
import matplotlib as nlp
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
import re
import string
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from math import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree
from sklearn.metrics import accuracy_score
from utils import clean_tweets, handle_emojis, evaluate_with_three_labels


tweets = pd.read_csv('tweets.csv')
tweets['sentiment'] = tweets['sentiment'].replace(['negative'],'0')
tweets['sentiment'] = tweets['sentiment'].replace(['positive'],'1')
tweets['sentiment'] = tweets['sentiment'].replace(['neutral'],'2')
print(tweets.head())
#Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) 
#that a search engine has been programmed to ignore,
#both when indexing entries for searching and when retrieving them as the result of a search query.

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopword = set(stopwords.words('english'))


tweets['cleaned_tweets']  = tweets['text'].apply(lambda x: clean_tweets(x))  


X = tweets.cleaned_tweets
y = tweets.sentiment
random_state = random.randint(10000,100000)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state = 100)
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


     
def decision_trees():
 
  clf = tree.DecisionTreeClassifier()
  clf.fit(X_train,y_train)
  y_pred=clf.predict(X_test)
  acc=accuracy_score(y_test,y_pred)
  return clf


evaluate_with_three_labels(decision_trees(), X_test, y_test)