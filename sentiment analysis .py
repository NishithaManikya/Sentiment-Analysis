#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import os


# ## Loading the data

# In[4]:


from nltk.corpus import stopwords
import string
 
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# In[5]:


filename = r'C:\Users\DELL\Downloads\txt_review\txt_reviews\review_10.txt'

text = load_doc(filename)

print(text)


# In[7]:


l=[]

for i in range(1,10000):
    filename=r'C:\Users\DELL\Downloads\txt_review\txt_reviews\review_{}.txt'.format(i)
    text = load_doc(filename)
    l.append(text)


# In[8]:


l[0]


# In[9]:


for i in range(0,len(l)):
    l[i]=l[i].split('\n')


# In[10]:


l[101][0]


# In[11]:


pid=[]
uid=[]
ProfileName=[]
HelpNumerator=[]
HelpDenominator=[]
Score=[]
Time=[]
ReviewSummary=[]
ReviewText=[]


for i in range(0,len(l)):
    pid.append(l[i][0])
    uid.append(l[i][1])
    ProfileName.append(l[i][2])
    HelpNumerator.append(l[i][3])
    HelpDenominator.append(l[i][4])
    Score.append(l[i][5])
    Time.append(l[i][6])
    ReviewSummary.append(l[i][7])
    ReviewText.append(l[i][8])


# ## Creating a data frame

# In[13]:


import pandas as pd
df=pd.DataFrame({'ProductId':pid,'UserId':uid, 'ProfileName':ProfileName, 'HelpfulnessNumerator':HelpNumerator,
                'HelpfulnessDenominator':HelpDenominator, 'Score':Score, 'Time': Time, 'ReviewSummary':ReviewSummary, 
                'ReviewText':ReviewText})


# In[14]:


df.head()


# ## Data Cleaning

# In[15]:


df['ProductId']=df['ProductId'].replace({'ProductId:':''},regex=True)
df['UserId']=df['UserId'].replace({'UserId:':''},regex=True)
df['ProfileName']=df['ProfileName'].replace({'ProfileName:':''},regex=True)
df['HelpfulnessNumerator']=df['HelpfulnessNumerator'].replace({'HelpfulnessNumerator:':''},regex=True)
df['HelpfulnessDenominator']=df['HelpfulnessDenominator'].replace({'HelpfulnessDenominator:':''},regex=True)
df['Score']=df['Score'].replace({'Score:':''},regex=True)
df['Time']=df['Time'].replace({'Time:':''},regex=True)
df['ReviewSummary']=df['ReviewSummary'].replace({'ReviewSummary:':''},regex=True)
df['ReviewText']=df['ReviewText'].replace({'ReviewText:':''},regex=True)


# In[17]:


df.head()


# In[18]:


df.tail()


# ## Exploratory Data Analysis

# In[19]:


df.isnull().sum()


# In[20]:


df.describe()


# In[21]:


df.dtypes


# In[22]:


df['HelpfulnessNumerator'] = df['HelpfulnessNumerator'].astype(str).astype('int64')


# In[23]:


df['HelpfulnessDenominator'] = df['HelpfulnessDenominator'].astype(str).astype('int64')


# In[24]:


df['Score'] = df['Score'].astype(str).astype('int64')


# In[25]:


review = df.drop_duplicates()


# In[26]:


review.info()


# In[27]:


df['usefulness'] = df['HelpfulnessNumerator']/df['HelpfulnessDenominator']


# In[28]:


usefulness = []
for i in df['usefulness']:
    if i > 0.75:
        usefulness.append(">75%")
    elif i < 0.25:
        usefulness.append("<25%")
    elif i >= 0.25 and i <= 0.75:
        usefulness.append("25-75%")
    else:
        usefulness.append("useless")
df['usefulness']  = usefulness   


# In[29]:


word_count = []
for i in df['ReviewText']:
    word_count.append(len(i.split()))
df['word_count'] = word_count    


# In[30]:


df.head()


# In[33]:


sentiment = []
for i in review['Score']:
    if i > 3:
        sentiment.append('positive')
    elif i < 3:
        sentiment.append('negative')
    else:
        sentiment.append('not')
review['sentiment'] = sentiment 


# In[34]:


review.head()


# In[35]:


positive = review[review['sentiment']=='positive']


# In[36]:


positive


# In[37]:


negative = review[review['sentiment']=='negative']


# In[38]:


negative 


# In[40]:


get_ipython().system('pip install wordcloud')


# In[42]:


get_ipython().system('pip install STOPWORDS')


# In[43]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd


# In[44]:


comment_words = ''
stopwords = set(STOPWORDS)

for val in positive['ReviewSummary']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "
    
wordcloud = WordCloud(width = 600, height = 600,
                background_color ='red',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)    
    
plt.figure(figsize = (10, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()      


# In[45]:


comment_words = ''
stopwords = set(STOPWORDS)

for val in negative['ReviewSummary']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "
    
wordcloud = WordCloud(width = 600, height = 600,
                #mask = mask,
                background_color ='yellow',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)    
    
plt.figure(figsize = (10, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()


# In[47]:


plt.figure(figsize=(10,5))
sns.countplot(x='Score',data=review)
plt.title(' Score(Ratings)')
plt.show()


# In[48]:


plt.figure(figsize=(10,8))
sns.countplot(x='usefulness',data=df,order=['useless','>75%','25-75%','<25%'],palette="rocket_r")
plt.title('Distribution of Helpfulness')
plt.show()


# In[49]:


df['usefulness'].value_counts()


# In[50]:


plt.figure(figsize=(10,8))
sns.countplot(x='Score',data=df,hue='usefulness',hue_order=['>75%','25-75%','<25%'],order=[5,4,3,2,1],palette="rocket_r")
plt.xticks(rotation=90,fontsize=10)
plt.title('COMPARING THE SCORE WITH USEFULNESS')
plt.show()


# In[51]:


plt.figure(figsize=(10,8))
sns.boxplot(x='Score',y='word_count',data=df,showfliers=False)
plt.xticks(rotation=90,fontsize=10)
plt.title('COMPARING THE SCORE WITH USEFULNESS')
plt.show()


# ## EDA Observations
#      1.5 Rating reviews are more and 2 Rating reviews are less
#      2.Useless reviews are more compared to useful reviews
#      3.Wordcount spread is less for 5 rating reviews.  

# ## Model Building

# ## Data Preprocessing

# In[53]:


total_size=len(df)
train_size=int(0.70*total_size)
train=df.head(train_size)
test=df.tail(total_size - train_size)


# In[54]:


train = train[train.Score != 3]
test = test[test.Score != 3]


# In[55]:


print(train.shape)
print(test.shape)


# In[56]:


train['Score'].value_counts()


# In[57]:


test['Score'].value_counts()


# In[58]:


lst_text = train['ReviewText'].tolist()
lst_summary = train['ReviewSummary'].tolist()


# In[59]:


lst_summary


# In[60]:


test_text = test['ReviewText'].tolist()
test_text


# ## Converting to lower case

# In[61]:


lst_text = [str(item).lower() for item in lst_text]
lst_summary = [str(item).lower() for item in lst_summary]


# In[62]:


test_text = [str(item).lower() for item in test_text]


# ## Removing the HTML Tags from the strings

# In[63]:


import re
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

for i in range(len(lst_text)):
    lst_text[i] = striphtml(lst_text[i])
    lst_summary[i] = striphtml(lst_summary[i])


# In[64]:


for i in range(len(test_text)):
    test_text[i] = striphtml(test_text[i])


# ## Removing the Special charecters from Strings

# In[65]:


for i in range(len(lst_text)):
    lst_text[i] = re.sub(r'[^A-Za-z]+', ' ', lst_text[i])
    lst_summary[i] = re.sub(r'[^A-Za-z]+', ' ', lst_summary[i])


# In[66]:


for i in range(len(test_text)):
    test_text[i] = re.sub(r'[^A-Za-z]+', ' ', test_text[i])


# ## Removing stop words

# In[67]:


import nltk
nltk.download('stopwords') 


# In[68]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# In[69]:


get_ipython().run_line_magic('time', '')
stop_words = set(stopwords.words('english'))
for i in range(len(lst_text)):
    text_filtered = []
    summary_filtered = []
    text_word_tokens = []
    summary_word_tokens = []
    text_word_tokens = lst_text[i].split()
    summary_word_tokens = lst_summary[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(r)
    lst_text[i] = ' '.join(text_filtered)
    for r in summary_word_tokens:
        if not r in stop_words:
            summary_filtered.append(r)
    lst_summary[i] = ' '.join(summary_filtered)


# In[70]:


for i in range(len(test_text)):
    text_filtered = []
    text_word_tokens = []
    text_word_tokens = test_text[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(r)
    test_text[i] = ' '.join(text_filtered)


# ## Stemming

# In[71]:


get_ipython().run_line_magic('time', '')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
for i in range(len(lst_text)):
    text_filtered = []
    summary_filtered = []
    text_word_tokens = []
    summary_word_tokens = []
    text_word_tokens = lst_text[i].split()
    summary_word_tokens = lst_summary[i].split()
    for r in text_word_tokens:
        text_filtered.append(str(stemmer.stem(r)))
    lst_text[i] = ' '.join(text_filtered)
    for r in summary_word_tokens:
        summary_filtered.append(str(stemmer.stem(r)))
    lst_summary[i] = ' '.join(summary_filtered)


# In[72]:


for i in range(len(test_text)):
    text_filtered = []
    text_word_tokens = []
    text_word_tokens = test_text[i].split()
    for r in text_word_tokens:
        if not r in stop_words:
            text_filtered.append(str(stemmer.stem(r)))
    test_text[i] = ' '.join(text_filtered)


# In[73]:


lst_text[0:5]


# ## Converting Text to Numerical vectors - BOW Representation

# In[75]:


from sklearn.feature_extraction.text import CountVectorizer
vocab = CountVectorizer()
train_bow = vocab.fit_transform(lst_text)


# In[76]:


train_bow


# In[77]:


X_test_dtm = vocab.transform(test_text)
X_test_dtm


# ## Multinomial Naive Bayes

# In[78]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(train_bow, train.Score)


# In[79]:


y_pred_class_nb = nb.predict(X_test_dtm)


# In[80]:


temp_df=pd.DataFrame({'Actual':test.Score,'Predicted': y_pred_class_nb})
temp_df.head()


# ## Evaluating NB

# In[81]:


from sklearn import metrics
from sklearn import metrics
nb_acc=metrics.accuracy_score(test.Score, y_pred_class_nb)
print(metrics.accuracy_score(test.Score, y_pred_class_nb))
print(metrics.classification_report(test.Score, y_pred_class_nb))
print(metrics.confusion_matrix(test.Score, y_pred_class_nb))


# In[82]:


sns.histplot(test.Score,color='blue', alpha=0.5)
sns.histplot(y_pred_class_nb, color='red',alpha=0.5)


# ## KNN

# In[84]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train_bow, train.Score)


# In[85]:


y_pred_class_knn = knn.predict(X_test_dtm)


# In[86]:


temp_df=pd.DataFrame({'Actual':test.Score,'Predicted': y_pred_class_knn})
temp_df.head()


# ## Evaluating KNN

# In[87]:


from sklearn import metrics
knn_accuracy=metrics.accuracy_score(test.Score, y_pred_class_knn)
print(metrics.accuracy_score(test.Score, y_pred_class_knn))
print(metrics.classification_report(test.Score, y_pred_class_knn))
print(metrics.confusion_matrix(test.Score, y_pred_class_knn))


# In[88]:


sns.histplot(test.Score,color='blue', alpha=0.5)
sns.histplot(y_pred_class_knn, color='red',alpha=0.5)


# ## Logistic Regression

# In[90]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_bow, train.Score)


# In[91]:


y_pred_class_logistic = classifier.predict(X_test_dtm)


# In[92]:


temp_df=pd.DataFrame({'Actual':test.Score,'Predicted': y_pred_class_logistic})
temp_df.head()


# ## Evaluating Logistic Regression

# In[94]:


log_acc=metrics.accuracy_score(test.Score, y_pred_class_logistic)
print(metrics.accuracy_score(test.Score, y_pred_class_logistic))
print(metrics.classification_report(test.Score, y_pred_class_logistic))
print(metrics.confusion_matrix(test.Score, y_pred_class_logistic))


# In[95]:


sns.histplot(test.Score,color='blue', alpha=0.5)
sns.histplot(y_pred_class_logistic, color='red',alpha=0.5)


# ## Decision Tree

# In[97]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
get_ipython().run_line_magic('time', 'tree.fit(train_bow, train.Score)')


# In[98]:


y_pred_class_tree = tree.predict(X_test_dtm)


# In[99]:


temp_df=pd.DataFrame({'Actual':test.Score,'Predicted': y_pred_class_tree})
temp_df.head()


# ## Evaluating Decision Tree

# In[101]:


dt_acc=metrics.accuracy_score(test.Score, y_pred_class_tree)
print(metrics.accuracy_score(test.Score, y_pred_class_tree))
print(metrics.classification_report(test.Score, y_pred_class_tree))
print(metrics.confusion_matrix(test.Score, y_pred_class_tree))


# In[102]:


sns.histplot(test.Score,color='blue', alpha=0.5)
sns.histplot(y_pred_class_tree, color='red',alpha=0.5)


# ### Random Forest

# In[104]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(train_bow, train.Score)


# In[105]:


y_pred_class_rf = rf.predict(X_test_dtm)


# In[106]:


temp_df=pd.DataFrame({'Actual':test.Score,'Predicted': y_pred_class_rf})
temp_df.head()


# ## Evaluating Random Forest Classifier

# In[108]:


rf_acc=metrics.accuracy_score(test.Score, y_pred_class_rf)
print(metrics.accuracy_score(test.Score, y_pred_class_rf))
print(metrics.classification_report(test.Score, y_pred_class_rf))
print(metrics.confusion_matrix(test.Score, y_pred_class_rf))


# In[109]:


sns.histplot(test.Score,color='blue', alpha=0.5)
sns.histplot(y_pred_class_rf, color='red',alpha=0.5)


# In[110]:


score= {'Algorithms':['Multinomial Naive Bayes','KNN','Desision tree','Logistic Regression', 'Random Forest Classifier'],
           'Scores':[nb_acc,knn_accuracy, dt_acc,log_acc, rf_acc]}
score_df= pd.DataFrame(score)
score_dfscore= {'Algorithms':['Multinomial Naive Bayes','KNN','Desision tree','Logistic Regression', 'Random Forest Classifier'],
           'Scores':[nb_acc,knn_accuracy, dt_acc,log_acc, rf_acc]}
score_df= pd.DataFrame(score)
score_df


# In[111]:


sns.barplot(data=score_df,x=score_df['Scores'],y=score_df['Algorithms'],)


# ## Random Forest and Logistic Regression has more accuracy

# In[ ]:




