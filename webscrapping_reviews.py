


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import time
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import requests


# In[2]:


wm_title=[]  
wm_date = []
wm_content = []
wm_rating = []

for i in range(1,150):

  link = "https://www.amazon.in/Whirlpool-SUPERB-ATOM-GREY-DAZZLE/product-reviews/B07WGD8QQT/ref=cm_cr_arp_d_viewopt_srt?ie=UTF8&reviewerType=all_reviews&sortBy=helpful&pageNumber="+str(i)
  response = requests.get(link)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 

  # extracting Review Title
  title = soup.find_all('a',class_='review-title-content')
  review_title = []
  for i in range(0,len(title)):
      review_title.append(title[i].get_text())
  review_title[:] = [titles.lstrip('\n') for titles in review_title]
  review_title[:] = [titles.rstrip('\n') for titles in review_title]
  wm_title = wm_title + review_title

  ## Extracting Ratings
  rating = soup.find_all('i',class_='review-rating')
  review_rating = []
  for i in range(2,len(rating)):
      review_rating.append(rating[i].get_text())
  #review_rating.pop(0)
  #review_rating.pop(0)
  review_rating[:] = [reviews.rstrip(' out of 5 stars') for reviews in review_rating]
  wm_rating = wm_rating + review_rating

  #Extracting Content of review
  review = soup.find_all("span",{"data-hook":"review-body"})
  review_content = []
  for i in range(0,len(review)):
      review_content.append(review[i].get_text())
  review_content[:] = [reviews.lstrip('\n') for reviews in review_content]
  review_content[:] = [reviews.rstrip('\n') for reviews in review_content]
  wm_content = wm_content + review_content

  #Extracting dates of reviews
  dates = soup.find_all('span',class_='review-date')
  review_dates = []
  for i in range(2,len(rating)):
      review_dates.append(dates[i].get_text())
  review_dates[:] = [reviews.lstrip('Reviewed in India on') for reviews in review_dates]
  #review_dates.pop(0)
  #review_dates.pop(0)
  wm_date  = wm_date + review_dates
  


# In[3]:


print(len(wm_title))
print(len(wm_rating))
print(len(wm_content))
print(len(wm_date))


# In[4]:


df = pd.DataFrame()
df['Title'] = wm_title
df['Ratings'] = wm_rating
df['Comments'] = wm_content
df['Date'] = wm_date

df.head(5)


# In[5]:


df.head(2)


# In[6]:


df['Date'] = pd.to_datetime(df['Date'])
df['Ratings'] = df['Ratings'].astype(float)
df.head(2)


# ## Text Cleaning
# 
# 1. lower the text
# 2. tokenize the text (split the text into words) and remove the punctuation
# 3. remove useless words that contain numbers
# 4. remove useless stop words like ‘the’, ‘a’ ,’this’ etc.
# 5. Part-Of-Speech (POS) tagging: assign a tag to every word to define 6. if it corresponds to a noun, a verb etc. using the WordNet lexical database
# 7. lemmatize the text: transform every word into their root form (e.g. rooms -> room, slept -> sleep)

# In[ ]:


df.head(2)


# In[7]:


from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# In[8]:


def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)


# In[9]:


# clean text data
df["Comments"] = df["Comments"].apply(lambda x: clean_text(x))


# In[10]:


df['Title'] = df['Title'].astype(str)
df['Title'] = df['Title'].apply(lambda x: clean_text(x))


# In[14]:


df.head(5)


# ## Feature Engineering

# In[23]:


#  add sentiment anaylsis columns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
df["sentiments"] = df["Comments"].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)
'''
sid = SentimentIntensityAnalyzer()
df["sentiments_title"] = df["Title"].apply(lambda x: sid.polarity_scores(x))
df = pd.concat([df.drop(['sentiments_title'], axis=1), df['sentiments_title'].apply(pd.Series)], axis=1)
'''
df


# In[12]:


# add number of characters column
df["nb_chars"] = df["Comments"].apply(lambda x: len(x))

# add number of words column
df["nb_words"] = df["Comments"].apply(lambda x: len(x.split(" ")))

''''
# add number of characters column
df["nb_chars_title"] = df["Title"].apply(lambda x: len(x))

# add number of words column
df["nb_words_title"] = df["Title"].apply(lambda x: len(x.split(" ")))
'''


# In[13]:


# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["Comments"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each Comment into a vector data
doc2vec_df = df["Comments"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
df = pd.concat([df, doc2vec_df], axis=1)

'''
# transform each Title into a vector data
doc2vec_df_title = df["Title"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df_title.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
df = pd.concat([df, doc2vec_df_title], axis=1)
'''


# In[14]:


# add tf-idfs columns
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(df["Comments"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df, tfidf_df], axis=1)

'''
##TF-IDF for Titles
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(df["Title"]).toarray()
tfidf_df_title = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df_title.columns = ["word_" + str(x) for x in tfidf_df_title.columns]
tfidf_df_title.index = df.index
df = pd.concat([df, tfidf_df_title], axis=1)
'''


# In[15]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(df["Comments"])


# In[16]:


# print wordcloud
show_wordcloud(df["Title"])


# In[21]:


df.head(2)


# In[17]:


# highest positive sentiment reviews (with more than 5 words)
df[df["nb_words"] >= 5].sort_values("pos", ascending = False)[["Comments", "pos"]].head(10)


# In[18]:


# lowest negative sentiment reviews (with more than 5 words)
df[df["nb_words"] >= 5].sort_values("neg", ascending = False)[["Comments", "neg"]].head(10)


# In[19]:


df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df.head(2)


# In[20]:


df_recent = df[(df['Year']== 2020) & (df['Month'] != 8)]
df_recent.head(2)


# In[33]:




