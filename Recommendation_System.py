#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[46]:


book=pd.read_excel(r'Book.xlsx')


# In[47]:


book.head()


# In[48]:


book.tail()


# In[49]:


book.shape


# In[50]:


book.info()


# In[51]:


book.isnull().sum()


# In[52]:


book.drop(book.columns[[0]],axis=1,inplace=True)


# In[9]:


book


# In[53]:


book.nunique()


# In[54]:


#Renaming the colums
book.columns = ["UserID","BookTitle","BookRating"]


# In[55]:


book


# In[56]:


book=book.sort_values('UserID')


# In[57]:


book


# In[58]:


#number of unique users in the dataset
len(book.UserID.unique())


# In[59]:


#Unique movies
book.BookTitle.unique()


# In[ ]:





# In[60]:


book.BookRating.value_counts()


# In[61]:


plt.figure(figsize=(20,6))
sns.distplot(book.BookRating)


# In[63]:


book_df = pd.pivot_table(book,index='UserID',columns='BookTitle', values='BookRating').reset_index(drop=True)


# In[64]:


book_df.fillna(0,inplace=True)


# In[65]:


book_df


# # AVERAGE RATING OF BOOK

# In[104]:


AVG = book['BookRating'].mean()
print(AVG)


# In[105]:


# Calculate the minimum number of votes required to be in the chart, 
minimum = book['BookRating'].quantile(0.90)
print(minimum)


# In[106]:


# Filter out all qualified Books into a new DataFrame
q_Books = book.copy().loc[book['BookRating'] >= minimum]
q_Books.shape


# # COSINE SIMILARITY BETWEEN USERS

# In[107]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[108]:


user_sim=1-pairwise_distances(book_df.values,metric='cosine')


# In[109]:


user_sim


# In[110]:


user_sim_df=pd.DataFrame(user_sim)


# In[111]:


user_sim_df


# In[112]:


#Set the index and column names to user ids 
user_sim_df.index = book.UserID.unique()
user_sim_df.columns = book.UserID.unique()


# In[113]:


user_sim_df


# In[114]:


np.fill_diagonal(user_sim,0)
user_sim_df


# In[117]:


#Most Similar Users
print(user_sim_df.idxmax(axis=1)[1378])
print(user_sim_df.max(axis=1).sort_values(ascending=False).head(50))


# In[118]:


reader = book[(book['UserID']==1378) | (book['UserID']==3192)]
reader


# In[119]:


reader1=book[(book['UserID']==1378)] 
reader1


# In[120]:


reader2=book[(book['UserID']==3192)] 
reader2


# In[ ]:




