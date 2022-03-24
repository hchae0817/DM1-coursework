# Part 3: Mining text data.

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
import numpy as np
import pandas as pd

def read_csv_3(data_file):
    data_file = pd.read_csv(data_file, encoding ='latin1')
    return data_file

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    arr = df["Sentiment"].to_numpy()
    unique_arr = np.unique(arr)
    return list(unique_arr)
    
# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    most = df['Sentiment'].value_counts()[1:2]
    return (most.index)[0]

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    dates = df.loc[df['Sentiment'] == 'Extremely Positive']
    dates_list = dates["TweetAt"].value_counts()
    return dates_list.index[0]
    
# Modify the dataframe df by converting all tweets to lower case.
def lower_case(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-zA_Z]', ' ', regex=True)

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    #df['OriginalTweet'] = df['OriginalTweet'].replace('\s+', ' ', regex=True)
    df['OriginalTweet'] = df['OriginalTweet'].str.strip()

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    df['OriginalTweet'] = df['OriginalTweet'].str.split()

# Given dataframe tdf with the tweets tokenized,
# return the number of words in all tweets including repetitions.

def count_words_with_repetitions(tdf):
    all_words = 0
    all_words = all_words + tdf['OriginalTweet'].apply(lambda x: len(x))
    return all_words.sum()

# Given dataframe tdf with the tweets tokenized,
# return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    all_words = tdf.explode('OriginalTweet')
    all_words_list = list(all_words['OriginalTweet'])
    all_words_no_rep = set(all_words_list)
    return len(all_words_no_rep)

# Given dataframe tdf with the tweets tokenized,
# return a list with the k distinct words that are most frequent in the tweets.

from collections import Counter

def frequent_words(tdf,k):
    all_words = tdf.explode('OriginalTweet')
    all_words_list = list(all_words['OriginalTweet'])
    freq_words = Counter(all_words_list).most_common(k)
    output = [c[0] for c in freq_words]
    return output

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt

import csv

def remove_stop_words(tdf):
    stop_words = pd.read_csv('https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt',
                     sep = " ", header = None, names = ['list'], quoting = csv.QUOTE_NONE, encoding = 'utf-8')
    
    for index,row in tdf['OriginalTweet'].items():
        tdf.at[index,'OriginalTweet'] = [x for x in row if (x not in stop_words) and len(x) > 2]

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.

from nltk.stem.porter import PorterStemmer
# Use English stemmer.
stemmer = PorterStemmer()

def stemming(tdf):
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [stemmer.stem(y) for y in x])
    
#[14 marks]

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	pass

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred,y_true):
	pass






