import pandas as pd
import numpy as np
import matplotlib as nlp
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
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
from utils import clean_tweets, handle_emojis, evaluate_with_three_labels, clean_dataset



#Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) 
#that a search engine has been programmed to ignore,
#both when indexing entries for searching and when retrieving them as the result of a search query.

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopword = set(stopwords.words('english'))

tweets = clean_dataset()
#print(tweets['cleaned_tweets'].head())

tokens=tweets['cleaned_tweets'].apply(lambda x: x.split())

token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts = cv.fit_transform(tweets['cleaned_tweets'].values.astype('U'))

X = text_counts
y = tweets['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=19)

def cnb():

  cnb = ComplementNB()
  cnb.fit(X_train, y_train)
  cross_cnb = cross_val_score(cnb, X, y,n_jobs = -1)
  print("Cross Validation score = ",cross_cnb)                
  print ("Train accuracy ={:.2f}%".format(cnb.score(X_train,y_train)*100))
  print ("Test accuracy ={:.2f}%".format(cnb.score(X_test,y_test)*100))
  train_acc_cnb=cnb.score(X_train,y_train)
  test_acc_cnb=cnb.score(X_test,y_test)
  return cnb


evaluate_with_three_labels(cnb(), X_test, y_test)