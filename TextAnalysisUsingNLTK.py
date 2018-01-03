# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 00:40:41 2017

@author: Vineet Garg -- DataPlayZ

@Topic : Exploring the 20 Newsgroups Dataset with Text Analysis Algorithim
"""

# Most Famous NLP library in python  is NLTK (others TextBlob, Gensim)
#Comes up with 50 collections large and well structured text datasets 
#which are called corpora in NLP, you can use import nltk, nltk.download() 
# select corpus of your choice

# Gensim is another powerful library
# Useful in Similarity querying, word vecorization, Distributed computing
# TextBlob is wrapper over NLTK and provide easy-to-use built-in functions 
#and methods 


from nltk.corpus import names
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# download the dataset
from sklearn.datasets import fetch_20newsgroups
# see the unique
import numpy as np


from sklearn.decomposition import NMF

# download the dataset
groups = fetch_20newsgroups()
ps = PorterStemmer()
lm = WordNetLemmatizer()

# lets check few names 
print(names.words()[:20])

# difference between Stemming and lemmatization is that lemmatization is a 
#cautious version of stemming
# examples 
ps.stem("machines")
ps.stem('learning')
# lemmatization algo based on wordnet corpus built-in
lm.lemmatize('machines')
lm.lemmatize('learning') # lm works on nouns not on verb

# let's have thinking about features
# you can check about datasets here
groups.keys()
groups['target']
groups['target_names']
np.unique(groups.target)
groups.data[0]
#answer yourself if len of doc is important feature here?

# Visualization
## It's a good to visualize to get the general idea of how the data is 
#sturctured, what possible issues may arise, and if there are any irregularities
# that we can take care of it, use seaborn package www.seaborn.pydata.org
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(groups.target)
plt.show()
# shows that distribution is fairly uniform, so that's one less thing to worry 
#about

# use countvectorizer class, refer help to get more about it
cv = CountVectorizer(stop_words="english",max_features = 500)
tf = cv.fit_transform(groups.data)
print(cv.get_feature_names())
# plot the top 500 fetaures
plt.xlabel('logcount')
plt.ylabel('Frequency')
plt.title('Distribution plot of top 500 words counts')
sns.distplot(np.log(tf.toarray().sum(axis=0)))
plt.show()

# now do some preprocessing, use some filtering mechnaism as top 500 words may 
# not contains english words and contains number which may be redundant for NLP

def letters_only(astr):
    return astr.isalpha()
cleaned = []
all_names = set(names.words())
# use lemmatizer
for post in groups.data:
    cleaned.append(' '.join([lm.lemmatize(word.lower())
    for word in post.split()
    if letters_only(word)
    and word not in all_names]))

tf = cv.fit_transform(cleaned)
print(cv.get_feature_names())
# plot the top 500 fetaures
plt.xlabel('logcount')
plt.ylabel('Frequency')
plt.title('Distribution plot of cleaned top 500 words counts')
sns.distplot(np.log(tf.toarray().sum(axis=0)))
plt.show()

#Clustering divides a dataset into clusters, many algos but we will use
# WSSSE i.e, within set sum of square errors or WCSS i.e. 
#Within cluster sum of squares
# Scikit-learn gives KMeans class  
from sklearn.cluster import KMeans 
# default n_clusters = 8, max_iter = 300, n_init = 10, tol = 1e-4
km = KMeans(n_clusters=20)
km.fit(tf)
labels = groups.target
plt.xlabel('Newsgroup')
plt.ylabel('Cluster')
plt.scatter(labels,km.labels_)


#Topic modeling : One common algo : non-negative matrix factorization
# factorize main matrix into 2 smaller matrix so there are no -ve number
# in all 3 metrices
nmf = NMF(n_components = 100, random_state=43).fit(tf)
for topic_idx, topic in enumerate(nmf.components_):
    label = '{}: '.format(topic_idx)
    print(label," ".join([cv.get_feature_names()[i]
        for i in topic.argsort()[:-9:-1]]))

# We have fundamental concepts pf NLP and realised some common
#tasks like using NLTK, topic modeling, tokenization tech, stemming, 
#lemmatization, learn about clustering, and NMF for topic modeling
    


