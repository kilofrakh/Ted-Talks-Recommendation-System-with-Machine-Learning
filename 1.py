import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
import string
import warnings
from scipy.stats import pearsonr
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
warnings.filterwarnings('ignore')


df = pd.read_csv('tedx_dataset.csv')
print(df.head())

df.shape

df.isnull().sum()

splitted = df['posted'].str.split(' ', expand=True)

# Creating columns for month and year of the talk
df['year'] = splitted[2].astype('int')
df['month'] = splitted[1]

df['year'].value_counts().plot.bar()
plt.show()

# Let's combine the title and the details of the talk.
df['details'] = df['title'] + ' ' + df['details']

# Removing the unnecessary information
df = df[['main_speaker', 'details']]
df.dropna(inplace = True)
df.head()

# We would like to have a copy of our data for future use.
data = df.copy()

def remove_stopwords(text):
  stop_words = stopwords.words('english')

  imp_words = []

  # Storing the important words
  for word in str(text).split():
    word = word.lower()
    
    if word not in stop_words:
      imp_words.append(word)

  output = " ".join(imp_words)

  return output

# Removing the stopwords
df['details'] = df['details'].apply(lambda text: remove_stopwords(text))
df.head()


punctuations_list = string.punctuation


def cleaning_punctuations(text):
    signal = str.maketrans('', '', punctuations_list)
    return text.translate(signal)


df['details'] = df['details'].apply(lambda x: cleaning_punctuations(x))
df.head()

details_corpus = " ".join(df['details'])

plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, 
               width=800,
               height=400).generate(details_corpus)
plt.axis('off')
plt.imshow(wc)
plt.show()

vectorizer = TfidfVectorizer(analyzer = 'word')
vectorizer.fit(df['details'])


def get_similarities(talk_content, data=df):

    # Getting vector for the input talk_content.
    talk_array1 = vectorizer.transform(talk_content).toarray()

    # We will store similarity for each row of the dataset.
    sim = []
    pea = []
    for idx, row in data.iterrows():
        details = row['details']

        # Getting vector for current talk.
        talk_array2 = vectorizer.transform(
            data[data['details'] == details]['details']).toarray()

        # Calculating cosine similarities
        cos_sim = cosine_similarity(talk_array1, talk_array2)[0][0]

        # Calculating pearson correlation
        pea_sim = pearsonr(talk_array1.squeeze(), talk_array2.squeeze())[0]

        sim.append(cos_sim)
        pea.append(pea_sim)

    return sim, pea
