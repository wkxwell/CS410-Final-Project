#==============================================
#CS410 2022 FINAL PROJECT 
#KAIXIN WANG / XINGCHEN WU
#==============================================

#code source 1
#Sentiment Analysis on twitter dataset | NLP
#https://www.kaggle.com/code/piyushagni5/sentiment-analysis-on-twitter-dataset-nlp

#code source 2
#Sentiment Analysis using Logistic Regression and Naive Bayes
#https://towardsdatascience.com/sentiment-analysis-using-logistic-regression-and-naive-bayes-16b806eb4c4b


import nltk                                  
from nltk.corpus import twitter_samples      
import matplotlib.pyplot as plt              
import numpy as np                           
#import customtkinter

import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples 

import re                                  
import string                              

from nltk.corpus import stopwords          
from nltk.stem import PorterStemmer        
from nltk.tokenize import TweetTokenizer   

from tkinter import *
import customtkinter


# download the stopwords for the process_tweet function
nltk.download('stopwords')

# download sample twitter dataset.
nltk.download('twitter_samples')
# select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# combine the positive tweets and negative tweets, where the positive comes first
tweets = all_positive_tweets + all_negative_tweets

# make a numpy array representing labels of the tweets, 
# 1 represents positive and 0 represents negative
labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

# Create two sub-datasets for both postive tweets and negative tweets
# One is for training and one is for testing
test_positive = all_positive_tweets[4000:]
train_positive = all_positive_tweets[:4000]
test_negative = all_negative_tweets[4000:]
train_negative = all_negative_tweets[:4000]

train_tweet = train_positive + train_negative
test_tweet = test_positive + test_negative

# combine positive and negative labels
train_label = np.append(np.ones((len(train_positive), 1)), np.zeros((len(train_negative), 1)), axis=0)
test_label = np.append(np.ones((len(test_positive), 1)), np.zeros((len(test_negative), 1)), axis=0)

# POS-TAGGING Tweet data
def process_tweet(tweet):
   
    stemmer= PorterStemmer()
    stopwords_english = stopwords.words('english')
    
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    clean_tweet = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            clean_tweet.append(stem_word)

    return clean_tweet

# build a dictionary to indicate each word (either a "pos" word or "neg" word)
# in the format (word,label) to its frequency.
def build_freqs(tweets, aol):
    
    # tweets: a list of tweets
    # aol: an m x 1 array with the sentiment label of each tweet (either 0 or 1)
   
    yslist = np.squeeze(aol).tolist()

    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    
    return freqs                 


# create frequency dictionary
freqs = build_freqs(train_tweet, train_label)


def sigmoid(z): 

    h = 1/(1+np.exp(-z))
   
    return h
def gradientDescent(x, y, theta, alpha, num_iters):
   
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        # getthe dot product of x and theta
        z = np.dot(x,theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))/m
        #print(J)

        # update the weights theta
        theta = theta - alpha*(np.dot(x.T,h-y))/m
        
    J = float(J)
    return J, theta

def extract_features(tweet, freqs):
    
    # tweet: string tweet data
    # freqs: a freqs dictionary mapping each (word,label)'s frequency
    
    # process_tweet tokenizes, stems, removes stopwords and punctuation
    word_l = process_tweet(tweet)
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
       
    # loop through each word in the list of words
    for word in word_l:
         
        if (word,1.0) in freqs:
            # increase the word count for the positive label 1
            x[0,1] += freqs[(word,1.0)]
        if(word,0.0) in freqs:
            # increase the word count for the negative label 0
            x[0,2] += freqs[(word,0.0)]
        
    return x

# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_tweet), 3))
for i in range(len(train_tweet)):
    X[i, :]= extract_features(train_tweet[i], freqs)
    

# training labels corresponding to X
Y = train_label

J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 3500)


def predict_tweet(tweet, freqs, theta):

    # extract the features of the tweet and store it into x
    x = extract_features(tweet, freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x,theta))
    
    return y_pred
    
def getAnalyzeResult(tweet):
    return float(predict_tweet(tweet, freqs, theta))

#UI using customkinter 
root = customtkinter.CTk()
root.title('CS410-Final Project: Kaixin Wang / XingChen Wu')
root.geometry("670x500")

def inputSentence():
    if(getAnalyzeResult(my_entry.get()) > 0.50):
        my_text.delete(1.0, END)
        my_text.insert(END, f' Analysze percentage - {getAnalyzeResult(my_entry.get())}\n\n')
        my_text.insert(END, f' Predicting Comment Sentiment: Positive :) \n\n')
    if(getAnalyzeResult(my_entry.get()) == 0.50):
        my_text.delete(1.0, END)
        my_text.insert(END, f' Analysze percentage - {getAnalyzeResult(my_entry.get())}\n\n')
        my_text.insert(END, f' Predicting Comment Sentiment: Neutral :o \n\n')
    if(getAnalyzeResult(my_entry.get()) < 0.50):
        my_text.delete(1.0, END)
        my_text.insert(END, f' Analysze percentage - {getAnalyzeResult(my_entry.get())}\n\n')
        my_text.insert(END, f' Predicting Comment Sentiment: Negative :( \n\n')


my_labelframe = customtkinter.CTkFrame(root, corner_radius=5)
my_labelframe.pack(pady=20)

#input for sentence 
my_entry = customtkinter.CTkEntry(my_labelframe, width=400, height=40, border_width=1, placeholder_text="Enter a sentence to analyze")
my_entry.grid(row=0, column=0, padx=10, pady=10)

my_button = customtkinter.CTkButton(my_labelframe, text="Analyze", command=inputSentence)
my_button.grid(row=0, column=1, padx=10)

text_frame = customtkinter.CTkFrame(root, corner_radius=5)
text_frame.pack(pady=10)

my_text = Text(text_frame, height=20, width=65, wrap=WORD, bd=0, bg="#292929", fg="silver")
my_text.pack(pady=10, padx=10)


root.mainloop()