from turtle import pos
import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest
from utils import clean_tweets

def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    shingles = text.lower()
    shingles = shingles.split()
    return shingles

db = pd.read_csv('tweets.csv')
db['cleaned_tweets']  = db['text'].apply(lambda x: clean_tweets(x)) 

def get_forest(df, perms):
    start_time = time.time()
    
    minhash = []
    
    #preprocess our tweet in shingles with the same format
    for text in df['cleaned_tweets']:
        shingles = preprocess(text)
        #print(shingles)
        #MinHash the sting on all our shingles in the string
        m = MinHash(num_perm=perms)
        for s in shingles:
            m.update(s.encode('utf8'))
            #store the minhash in a string
        minhash.append(m)
    #build a forest of all the MinHashed strings    
    forest = MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
    #index our forest to make it searchable    
    forest.index()
    
    print('It took %s seconds to build forest.' %(time.time()-start_time))
    
    return forest


def predict(text, df, permutations, num_results, forest):
    start_time = time.time()
    #preprocess our text in shingles
    shingles = preprocess(text)
    #set the same number of permutations for your MinHash as used in building the forest
    m = MinHash(num_perm=permutations)
    #create the MinHash on text using all of our shingles
    for s in shingles:
        m.update(s.encode('utf8'))
    #query the forest with our MinHash and return the number of needed reccomenations    
    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None # if your query is empty, return none
    
    #create a list with the tweets and its sentiment
    #result = list(zip(df.iloc[idx_array]['sentiment'],df.iloc[idx_array]['cleaned_tweets']))
    result = df.iloc[idx_array]['sentiment']
    #convert the list in a df for better showing results, containing the sentiment and tweet
    #res = pd.DataFrame(result, columns=['Sentiment','Tweet'])
    #res = pd.DataFrame(result, columns=['Sentiment'])
    
   # print('It took %s seconds to query forest.' %(time.time()-start_time))
    
    return result


def print_results(keyword):
    permutations = 128

    forest = get_forest(db, permutations)
    #We set the number of recommendations that we want
    num_recommendations = 10
    #We set the keyword to find those recommandations as keyword
    #keyword = 'Today was a good day!'
    #print(predict(keyword, db, permutations, num_recommendations, forest))
    result = len(predict(keyword, db, permutations, num_recommendations, forest))
    res_pos = (predict(keyword, db, permutations, num_recommendations, forest))
    res_neg = (predict(keyword, db, permutations, num_recommendations, forest))
    res_neu =  (predict(keyword, db, permutations, num_recommendations, forest))
    res_pos=res_pos.to_frame()
    res_neg=res_neg.to_frame()
    res_neu=res_neu.to_frame()
    res_pos_total = res_pos[res_pos['sentiment'] == 'positive'].count()
    res_neg_total = res_neg[res_neg['sentiment'] == 'negative'].count()
    res_neu_total = res_neu[res_neu['sentiment'] == 'neutral'].count()
    pos_sentiment = res_pos_total/result
    neg_sentiment = res_neg_total/result
    neu_sentiment = res_neu_total/result
    return [pos_sentiment, neg_sentiment, neu_sentiment]
    #print('The tweet', keyword,'is positive with probability:', res_pos_total/result)
    #print('The tweet', keyword,'is negative with probability:', res_neg_total/result)
    #print('The tweet', keyword,'is neutral with probability:', res_neu_total/result)
    #print("Res pos total:", res_pos_total, "\n resneg:", res_neg_total, "\n res neu:", res_neu_total)

