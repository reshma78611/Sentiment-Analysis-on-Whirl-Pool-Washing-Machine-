# Sentiment Analysis on Whirl Pool Washing Machine Amazon Reviews

## Overview
In this  Sentiment Analysis is performed on Reviews given by customers in Amazon website for the product *Whirl Pool Waching Machine*

## Motivation
What could be a perfect way to utilize unfortunate lockdown period? Like most of you, I spend my time in cooking, coding and reading some latest research papers on weekends. So i got more interest on Data Science,and i have started research on that and learnt Data Science and to explore more i started Projects on ML and NLP. This is one of my simple NLP projects *Sentiment analysis on Amazon Reviews of Whirl Pool Washing Machine*

## Technical Aspect:
I have selected randomly a product from amazon that is Whirl Pool Washing Machine

This project is divided into three parts:

**1. Extracting information from Website:**

   Extracting information from Amazon using Beautiful soup on which we need to perform Sentiment Analysis, Information such as *Review Title, Review Ratings, Review Content, Review Date* and are stored as list.
   
   Now store each of information as feature and create  a dataframe
   
**2. Text Cleaning:**
   
   Since we have got all text data in the form of a dataframe, Now we will perform Text Cleaning on data such as 
   
      - Tokenization — convert sentences to words
      - Removing unnecessary punctuation, tags
      - Lowering the text
      - Removing stop words — frequent words such as ”the”, ”is”, etc. that do not have specific semantic
      - Stemming — words are reduced to a root by removing inflection through dropping unnecessary characters, usually a suffix.
      - Lemmatization — Another approach to remove inflection by determining the part of speech and utilizing detailed database of the language.
      - Bag of Words
      - TF-IDF
     
**3. Sentiment Analysis:**
    
   Since we have got all the cleaned data we will perform Sentiment Analysis using *Sentiment Intensity Analyzer* , with this we can extract pos, neg, neu, compound values. Now we can apply doc2vec and TFIDF Vectorizer and add features such as numbr of words and number of characters. Once its done sort the top positive words and top negative wordswith respect to pos and neg values. 
      
## Installation
    This Code is written in Python 3.7
