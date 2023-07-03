#!/usr/bin/env python
# coding: utf-8

# In[182]:


import numpy as np
import pandas as pd


# In[183]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[184]:


movies.head(1)


# In[185]:


credits.head(1)


# # Merge Datasets On The Basis Of Same Column

# In[186]:


movies=movies.merge(credits,on='title')


# In[187]:


movies.head(1)


# In[188]:


movies.info()


# In[189]:


#genres
#id
#keywords
#title
#overview
#cast                  
#crew                  

movies=movies[['genres','id','keywords','title','overview','cast','crew' ]]


# In[190]:


movies.info()


# In[191]:


movies.head()


# # missing data

# In[192]:


movies.isnull().sum()


# In[193]:


movies.dropna(inplace=True)


# In[194]:


movies.isnull().sum()


# In[195]:


movies.duplicated().sum()


# In[196]:


movies.iloc[0].genres


# In[197]:


import ast
ast.literal_eval


# In[198]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[199]:


movies['genres']= movies['genres'].apply(convert)


# In[200]:


movies.head()


# In[201]:


movies['keywords']= movies['keywords'].apply(convert)


# In[202]:


movies.head()


# In[203]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter += 1
        else:
            break 
    return L


# In[204]:


movies['cast'] = movies['cast'].apply(convert3)


# In[205]:


movies.head(1)


# In[206]:


movies['crew'][0]


# In[207]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] =='Director':
            L.append(i['name'])
            break 
    return L


# In[208]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[209]:


movies.head(1)


# In[210]:


movies['overview'][0]


# In[211]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[212]:


movies.head(1)


# In[213]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])


# In[214]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])


# In[215]:


movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])


# In[216]:


movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[217]:


movies.head(1)


# In[218]:


movies['tags']= movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']+ movies['overview']


# In[219]:


movies.head(1)


# In[220]:


new_df = movies[['id','title','tags']]


# In[221]:


new_df


# In[222]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[223]:


new_df['tags'][0]


# In[224]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[225]:


new_df.head()


# In[226]:


get_ipython().system('pip install nltk')


# In[227]:


import nltk


# In[228]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[229]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[230]:


new_df['tags']=new_df['tags'].apply(stem)


# In[231]:


len(cv.get_feature_names())


# In[232]:


ps.stem('danced')


# In[233]:


stem('Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization.')


# In[234]:


cv.get_feature_names()


# In[235]:


vectors


# # Cosine Distance

# In[236]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[237]:


vector = cv.fit_transform(new_df['tags']).toarray()


# In[238]:


vector.shape


# In[239]:


from sklearn.metrics.pairwise import cosine_similarity


# In[240]:


similarity = cosine_similarity(vector)


# In[241]:


similarity


# In[243]:


new_df[new_df['title'] == 'The Lego Movie'].index[0]


# In[244]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
        


# In[245]:


recommend('Gandhi')


# In[249]:


recommend('Avatar')


# In[246]:


import pickle


# In[252]:


new_df['title'].values


# In[256]:


import pandas as pd
pd.read_pickle("movies.pkl")


# In[257]:


new_df.to_dict()


# In[258]:


pickle.dump(new_df,open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[259]:


pickle.dump(new_df.to_dict,open('movies_dict.pkl','wb'))


# In[260]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[ ]:




