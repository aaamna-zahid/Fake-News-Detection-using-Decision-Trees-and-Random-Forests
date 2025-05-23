#!/usr/bin/env python
# coding: utf-8

# Fake news detection

# In[160]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from sklearn.metrics import accuracy_score, classification_report  # Import metrics for evaluation



# Read datasets

# In[162]:


fake = pd.read_csv("C:/Users/HP/Downloads/Fake.csv")
true = pd.read_csv("C:/Users/HP/Downloads/True.csv")


# Data cleaning and preparation

# In[164]:


fake['target'] = 'fake'
true['target'] = 'true'


# In[271]:


fake.head()


# In[191]:


# Limit the dataset to the first 21,417 rows
fake = fake.iloc[:21417]

# Save the truncated dataset to a new CSV file (optional)
fake.to_csv("fake.csv", index=False)


# In[273]:


true.head()


# In[193]:


# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop = True)
data.shape


# In[197]:


data.head(5)


# In[199]:


data.tail(5)


# In[201]:


# Shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)


# In[203]:


# Check the data
data.head()


# In[205]:


data.info()


# In[207]:


# Removing the date 
data.drop(["date"],axis=1,inplace=True)
data.head()


# In[209]:


# Removing the title
data.drop(["title"],axis=1,inplace=True)
data.head()


# In[211]:


# Convert to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())
data.head()


# In[212]:


# Remove punctuation

import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)


# In[214]:


# Check
data.head()


# In[215]:


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[216]:


data.head()


# Basic data exploration

# In[218]:


# How many articles per subject?
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()


# In[219]:


# How many fake and real articles?
print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()


# In[220]:


# Word cloud for fake news
from wordcloud import WordCloud

fake_data = data[data["target"] == "fake"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[221]:


# Word cloud for real news
from wordcloud import WordCloud

real_data = data[data["target"] == "true"]
all_words = ' '.join([text for text in fake_data.text])

wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)

plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[222]:


# Most frequent words counter   
from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


# In[223]:


# Most frequent words in fake news
counter(data[data["target"] == "fake"], "text", 20)


# In[224]:


# Most frequent words in real news
counter(data[data["target"] == "true"], "text", 20)


# In[225]:


# Function to plot the confusion matrix
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Split Data

# In[227]:


# Split the data
X_train,X_test,y_train,y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)


# In[228]:


X_train.head()


# In[229]:


y_train.head()


# Decision Tree Classifier

# In[269]:


# Decision Tree
dt_pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('model', DecisionTreeClassifier(criterion='gini', max_depth=None, splitter='best', random_state=0))])
dt_model = dt_pipe.fit(X_train, y_train)
dt_prediction = dt_model.predict(X_test)

# Random Forest
rf_pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('model', RandomForestClassifier(n_estimators=200, max_depth=None, random_state=0))])
rf_model = rf_pipe.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)

# Evaluation
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_prediction))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_prediction))

# Accuracy Comparison
print("Decision Tree accuracy: {}%".format(round(accuracy_score(y_test, dt_prediction) * 100, 2)))
print("Random Forest accuracy: {}%".format(round(accuracy_score(y_test, rf_prediction) * 100, 2)))

# Confusion Matrices
dt_cm = metrics.confusion_matrix(y_test, dt_prediction)
plot_confusion_matrix(dt_cm, classes=['Fake', 'Real'], title='Decision Tree Confusion Matrix')

rf_cm = metrics.confusion_matrix(y_test, rf_prediction)
plot_confusion_matrix(rf_cm, classes=['Fake', 'Real'], title='Random Forest Confusion Matrix')


# In[277]:


def check_news_validity(news_texts):
    final_predictions = []
    
    for news_text in news_texts:
        # Data cleaning for each input news
        news_text = news_text.lower()
        news_text = punctuation_removal(news_text)
        news_text = ' '.join([word for word in news_text.split() if word not in stop])
        
        # Predict using Decision Tree
        dt_prediction = dt_model.predict([news_text])[0]
        
        # Predict using Random Forest
        rf_prediction = rf_model.predict([news_text])[0]
        
        # Get final prediction
        final_prediction = "True" if dt_prediction == 'true' else "Fake"
        
        # Append the result
        final_predictions.append(final_prediction)

    return final_predictions

# Example usage with multiple inputs:
news_inputs = [
    "WASHINGTON (Reuters) - The White House is willing to consider a small increase on the corporate tax rate if it is needed to finalize the bill in the U.S. Congress, White House budget chief Mick Mulvaney said on Sunday. Mulvaney made his comments after President Donald Trump suggested on Saturday that the corporate tax rate could end up at 22 percent once the Senate and House of Representatives reconcile or â€œconferenceâ€ their respective versions of the legislation, even though both bills currently stand at 20 percent. â€œMy understanding is that the Senate (bill) has a 20 percent rate now. The House has a 20 percent rate now. Weâ€™re happy with both of those numbers,â€ Mulvaney said in an interview with CBSâ€™ â€œFace the Nation.â€ â€œIf something small happens in conference that gets us across the finish line, weâ€™ll look at it on a case-by-case basis. But I donâ€™t think youâ€™ll see any significant change in our position on the corporate taxes,â€ Mulvaney said. ",
    "Donald Trump signed an executive order withdrawing the US from the Trans-Pacific Partnership agreement in early 2017",
    "The Supreme Court has ruled in favor of climate change regulations, requiring companies to reduce emissions by 40% over the next decade.",
    "WEST PALM BEACH, Fla (Reuters) - President Donald Trump said on Thursday he believes he will be fairly treated in a special counsel investigation into Russian meddling in the U.S. presidential election, but said he did not know how long the probe would last. The federal investigation has hung over Trumpâ€™s White House since he took office almost a year ago, and some Trump allies have in recent weeks accused the team of Justice Department Special Counsel Robert Mueller of being biased against the Republican president. But in an interview with the New York Times, Trump appeared to shrug off concerns about the investigation, which was prompted by U.S. intelligence agenciesâ€™ conclusion that Russia tried to help Trump defeat Democrat Hillary Clinton by hacking and releasing embarrassing emails and disseminating propaganda. â€œThereâ€™s been no collusion. But I think heâ€™s going to be fair,â€ Trump said in what the Times described as a 30-minute impromptu interview at his golf club in West Palm Beach, Florida. Mueller has charged four Trump associates in his investigation. Russia has denied interfering in the U.S. election. U.S. Deputy Attorney General Rod Rosenstein said this month that he was not aware of any impropriety by Muellerâ€™s team. Trumpâ€™s lawyers have been saying for weeks that they had expected the Mueller investigation to wrap up quickly, possibly by the end of 2017. Mueller has not commented on how long it will last. Trump told the Times that he did not know how long the investigation would take. â€œTiming-wise, I canâ€™t tell you. I just donâ€™t know,â€ he said. Trump said he thought a prolonged probe â€œmakes the country look badâ€ but said it has energized his core supporters. â€œWhat itâ€™s done is, itâ€™s really angered the base and made the base stronger. My base is strong than itâ€™s ever been,â€ he said. The interview was a rare break in Trumpâ€™s Christmas vacation in Florida. He has golfed each day aside from Christmas Day, and mainly kept a low profile, apart from the occasional flurry of tweets. He spent one day golfing with Republican Senator David Perdue from Georgia, who has pushed legislation to cap immigration numbers, and had dinner on Thursday with Commerce Secretary Wilbur Ross, an international trade hawk. Trump told the Times he hoped to work with Democrats in the U.S. Congress on a spending plan to fix roads and other infrastructure, and on protections for a group of undocumented immigrants who were brought to the United States as children. Trump spoke about trade issues, saying he had backed off his hard line on Chinese trade practices in the hope that Beijing would do more to pressure North Korea to end its nuclear and missile testing program. He said he had been disappointed in the results. He also complained about the North American Free Trade Agreement (NAFTA), which his administration is attempting to renegotiate in talks with Mexico and Canada. Trump said Canadian Prime Minister Justin Trudeau had played down the importance of Canadian oil and lumber exports to the United States when looking at the balance of trade between the two countries. â€œIf I donâ€™t make the right deal, Iâ€™ll terminate NAFTA in two seconds. But weâ€™re doing pretty good,â€ Trump said. "
]

results = check_news_validity(news_inputs)
print(results)


# In[ ]:




