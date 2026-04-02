from flask import Flask,render_template
import pandas as pd 
import numpy as np 
import warnings
import nltk
import nltk.tokenize 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import matplotlib.pyplot as plt 
import re
from sklearn.feature_extraction.text  import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')


warnings.filterwarnings('ignore')

df = pd.read_csv('Restaurant_Reviews.tsv',sep='\t')

df['char_count']=df['Review.'].apply(len)
df['words_count']=df['Review.'].apply(lambda x:len(str(x).split()))
df['Sentence_count'] = df['Review.'].apply(lambda x :len(nltk.sent_tokenize(str(x))))


avg_char_count1=df[df['Liked']==1]['char_count'].mean()
avg_char_count0 = df[df['Liked']==0]['char_count'].mean()

review=re.sub('[^a-zA-Z]',' ',df['Review.'][1])
review=review.lower().split()  

all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
review = [word for word in review if word not in all_stopwords]

ps = PorterStemmer()
#review = [ps.stem(word) for word in review]
#review=" ".join(review)

custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                    'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

stop_words= set(stopwords.words('english'))-custom_stopwords
corpus = []
for i in range(len(df)):
    review=re.sub('[^a-zA-Z]',' ',df['Review.'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus.append(review)

df['processed_text'] = corpus

wc = WordCloud(width=500,height=500,min_font_size=8,background_color="white")
positive = wc.generate(df[df['Liked']==1]['processed_text'].str.cat(sep=" "))


negetive = wc.generate(df[df['Liked']==0]['processed_text'].str.cat(sep=" "))


cv = CountVectorizer(max_features=1500)

x = cv.fit_transform(corpus).toarray()
y = df['Liked']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc_y_prediction = rfc.predict(x_test)

accuracy3 = accuracy_score(y_test,rfc_y_prediction)
print(accuracy3) #accuracy: 0.8 best accuracy


import pickle
pickle.dump(rfc , open('model.pkl','wb'))
pickle.dump(cv, open('cv.pkl','wb'))