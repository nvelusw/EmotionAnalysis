#!/usr/bin/env python
# coding: utf-8

# In[2]:


import csv
import json
import regex as regularExpression
import pandas as pd
import numpy as np
import string
import demoji
import emoji
demoji.download_codes()
import re
import nltk
nltk.download('sentiwordnet')
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn


# In[3]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
from nltk import TweetTokenizer
from nltk import pos_tag,pos_tag_sents
#nltk.download('averaged_perceptron_tagger')
#nltk.download()
from nltk.corpus import wordnet as wn


# In[4]:


#Reading data into a dataframe
data_read = pd.read_csv('C:\\Users\\nandh\\OneDrive\\Desktop\\PDB\\Data\\test.csv',encoding='utf-8')
data_test = pd.read_csv('C:\\Users\\nandh\\OneDrive\\Desktop\\PDB\\Data\\train.csv',encoding='utf-8')


# In[5]:


data_read['Tweets']


# In[6]:


# List containing short words and its corresponding english words
contractionsDictionary = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"i'd": "I would",
"i'd've": "I would have",
"i'll": "I will",
"i'll've": "I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that had",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"thx": "thanks",
"thxs": "thanks",
"r":"are",
"u":"you",
"ur":"your",
"haha":"happy",
"ya":"yes",
"thanx":"thanks",
"plz":"please",
"pls":"please",
"k":"ok"
}


# In[7]:


# funtion to preprocess the given text to remove punctuations, remove html tags, number and conversion of text to lower case
def preprocess(text):
    text=str(text)
    text = text.lower()
    text=text.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', text)
    rem_url=re.sub(r'http\S+', '',cleantext)
    clean_data = re.sub('[0-9]+', '', rem_url)
    clean_data = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",clean_data)
    clean_data = re.sub("(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",clean_data)
    return clean_data


# In[8]:


#Applying preprocess function
data_read['clean_Text']=data_read['Tweets'].apply(lambda s:preprocess(s))
data_test['clean_Text']=data_test['Tweets'].apply(lambda s:preprocess(s))


# In[9]:


data_read.iloc[:,0:3]


# In[10]:


#Function to tokenize words after preprocessing
def shortToFullText(shorttext):
    ProcessedString = shorttext.copy()
    for txt in ProcessedString:
        if txt in contractionsDictionary.keys():
            loc = ProcessedString.index(txt)
            ProcessedString[loc]=contractionsDictionary[txt]
        continue
    return ProcessedString


# In[11]:


#Function to tokenize words after preprocessing
def tokenize(text):
    tokenizer=TweetTokenizer()
    t_token = tokenizer.tokenize(text)
    return t_token


# In[12]:


#Applying tokenization function
data_read['Tokenized']=data_read['clean_Text'].apply(lambda s:tokenize(s))
data_test['Tokenized']=data_test['clean_Text'].apply(lambda s:tokenize(s))


# In[13]:


data_read.iloc[:,2:4]


# In[14]:


#HappyEmoticons
emoticons_happy = [
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3','(:'
    ]


# In[15]:


# Sad Emoticons
emoticons_sad = [
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ]


# In[16]:


# Converting emoji_symbols to text
def emoji_to_text(txts):
    emoji_list=txts.copy()
    for i in emoji_list:
        for j in emoticons_happy:
            re.sub(' +', ' ', i).strip()
            if j == i:
                loc = emoji_list.index(i)
                emoji_list[loc]=":Happy_emoji:"
        continue
        for k in emoticons_sad:
            if k == i:
                location = emoji_list.index(i)
                emoji_list[location]=":Sad_emoji:"
        continue
    return emoji_list


# In[17]:


#Applying emoji_to_text function
data_read['emoji_conversion']=data_read['Tokenized'].apply(lambda s:emoji_to_text(s))
data_test['emoji_conversion']=data_test['Tokenized'].apply(lambda s:emoji_to_text(s))


# In[18]:


data_read.iloc[8:45,3:5]


# In[19]:


#Converting emoticons to text
def emoticons_to_Text(words):
    convertedlist=words.copy()
    for txt in convertedlist:
        str=emoji.demojize(txt)
        loc = convertedlist.index(txt)
        convertedlist[loc]=str
    return convertedlist


# In[20]:


#Applying emoticon function
data_read['emoji_list']=data_read['emoji_conversion'].apply(lambda s:emoticons_to_Text(s))
data_test['emoji_list']=data_test['emoji_conversion'].apply(lambda s:emoticons_to_Text(s))


# In[21]:


data_read.iloc[8:45,4:6]


# In[22]:


#denoising
def denoising_characters(words):
    denoise=words.copy()
    spec_chars=[":","?","<",">",".","/","*","#","!","\,","\\","|","\'","\"","[","]","{","}","(",")","+","-","_","&","^","$","%","~","`","@",","," ?"," !"]
    for word in list(spec_chars):
        re.sub(' +', ' ', word).strip()
        re.sub(r'\s([?.!"](?:\s|$))', r'\1', word)
        if ((word in denoise) and (len(word)==1)):
            denoise.remove(word)
    #denoise = [word for word in denoise if word.isalpha()]
    return denoise


# In[23]:


#Applying denoising function
data_read['post_denoise']=data_read['emoji_list'].apply(lambda s:denoising_characters(s))
data_test['post_denoise']=data_test['emoji_list'].apply(lambda s:denoising_characters(s))


# In[24]:


data_read.iloc[8:45,5:7]


# In[25]:


# POS tagger function
def posTaggerFunction(texts):
    posConversionList=[]
    for word in texts:
        posConversionList.append(nltk.tag.pos_tag(word_tokenize(word)))  
    return posConversionList


# In[26]:


#Removing stop words
def stop_words(txts):
    to_be_list=txts.copy()
    stop_words=['those', 'they', 'further', 'most', 'or', 'whom', "you'd", 'until', 'shan', 'few', 'in', 'more', 'again', 'against', 'myself', "haven't", 'at', 'where', 'how', 'than', 's', 'are', 'before', 'll', 'wouldn', 'and', 'won', 'your', 'both', 'what', 'an', 'you', 'any', 'd', 'being', 'the', 'out', 'should', 'o', 'own',  "you're", 'had', 'my', 'this', 'to', 'ours', 'then', 'him', 'it', 'off', 'each', 'ain', 'into',  'does', 'because', 'above', 'as', 'ma', 'once', 'having', 'which', 'below', 'by', 'up', 'over','hmm', 'were', "that'll", 'can', 'here', "you've", 'themselves', "mightn't", 'such', 'me', 'through', "should've", 'did', 'on', 'needn', 'their', 'itself', 'am', "hadn't", 'of', 'have', 'we', 'theirs', "hasn't", 'only', 'he', 'm', 're', 'i', 'who', "she's", 'just', 'ourselves', "you'll", 'mightn', 'her', 'been', 'aren', 'with',  'herself', 'y', 'hers', 'do', 'weren', 'yourself', 'so', 've', 'his', 'these', 'mustn', 'a', 'about', 'all', 'yourselves', 'under', 'there', 'same', 'them', 'now', 'will', 'our', 'that',  'some', 'doing', 'but', 'himself', 'when', 'is', 'why', 'for',  'its', 'has',  'other',  "it's", 'yours', 'be', 'was', 'during',  'between', 'she', 'while', 'from', 'if','ur','gf' 'after','!','@','#','"','$','(','.',')','u','r','i']
    for txt in list(to_be_list):
        txt.strip()
        if txt in stop_words:
            to_be_list.remove(txt)
    return to_be_list


# In[27]:


# Applying shortext to full text function
data_read['post_shorttext']=data_read['post_denoise'].apply(lambda s:shortToFullText(s))
data_test['post_shorttext']=data_test['post_denoise'].apply(lambda s:shortToFullText(s))


# In[28]:


data_read.iloc[8:45,6:8]


# In[29]:


#Applying stopwords function
data_read['post_Stop']=data_read['post_shorttext'].apply(lambda s:stop_words(s))
data_test['post_Stop']=data_test['post_shorttext'].apply(lambda s:stop_words(s))


# In[30]:


data_read.iloc[8:45,6:9]


# In[31]:


#Removing emoticons seperately
def emojisepearation(sentence):
    to_be_extracted=sentence.copy()
    for text in to_be_extracted:
        #to_be_extracted=(re.match(r':a-zA-Z:'),text)
        filtered = filter(re.match(r':\w+:'), text) 
    if filtered!=0:
        return filtered
    else: 
        return None


# In[32]:


#Applying posTagger function
data_read['post_posConversion']=data_read['post_Stop'].apply(lambda s:posTaggerFunction(s))


# In[33]:


data_read['post_posConversion']


# In[34]:


# function to count the occurence of words
#global word_frequency
word_frequency={}
def frequent_occuringWords(words):
    for element in words:
        if element in word_frequency:
            word_frequency[element] = word_frequency[element] + 1
        else:
            word_frequency[element] = 1


# In[35]:


#Applying frequent_occuringWords function
data_read['post_Stop'].apply(lambda s:frequent_occuringWords(s))


# In[36]:


#sorting the frequently occuring words
from collections import OrderedDict
frequent_words = sorted(word_frequency.items(), key=lambda kv: kv[1], reverse=True)
print(frequent_words)


# In[37]:


#Feature Exctraction extracting 100 frequently occuring words
result_list =[x[0] for x in frequent_words]
toplist=result_list[:500]
print(toplist)


# In[38]:


russel_complex_list=['alarmed','tense','afraid','angry','annoyed','distressed','frustrated','aroused','astonished','excited','delighted','happy','pleased','glad','serene','content','atease','satisfied','relaxed','calm','miserable','sad','gloomy','depressed','bored','droopy','tired']


# In[39]:


#Extracted Knowledge Base
seedlist=list(toplist) 
seedlist.extend(element for element in russel_complex_list if element not in seedlist) 
print(seedlist)


# In[40]:


def tag_conversion(to_be_convertedtag):
    if to_be_convertedtag.startswith('J'):
        return wn.ADJ
    elif to_be_convertedtag.startswith('N'):
        return wn.NOUN
    elif to_be_convertedtag.startswith('R'):
        return wn.ADV
    elif to_be_convertedtag.startswith('V'):
        return wn.VERB
    return None


# In[41]:


from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
seedlist_Classification={}
def word_classification(frequent_list):
    for word in frequent_list:
        words = word_tokenize(word)
        tags = pos_tag(words)
        wn_tag = tag_conversion(tags[0][1])
        word = tags[0][0]
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize(word)
        if not lemma:
            continue
        synsets = wn.synsets(lemma, pos=wn_tag)
      #  print(synsets)
        if not synsets:
            continue
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        positive_score= float(swn_synset.pos_score())
        negative_score= float(swn_synset.neg_score())
        if(positive_score > negative_score):
            if(positive_score >= 0.5 and positive_score < 1 ):
                seedlist_Classification[word]="Happy-Active Class"
            elif(positive_score >= 0.1 and positive_score < 0.5 ):
                seedlist_Classification[word]="Happy-Inactive Class"
        elif (positive_score < negative_score):
            if(negative_score >= 0.5 and negative_score < 1 ):
                seedlist_Classification[word]="Unhappy-Active Class"
            elif(negative_score >= 0.1 and negative_score < 0.5 ):
                seedlist_Classification[word]="Unhappy-Inactive Class"
        elif (positive_score == negative_score):
            continue 
        


# In[42]:


word_classification(seedlist)


# In[43]:


print(seedlist_Classification)


# In[44]:


#rule-based classification
def ruleBasedClassification(tagged_words):
    taglist=[]
    classified_class_count={}
    for txt in tagged_words:
        for word, tag in txt:
            tuplelist=[]            
            if word in seedlist_Classification.keys():
                classified_class=seedlist_Classification[word]
                if classified_class in classified_class_count:
                    classified_class_count[classified_class] = classified_class_count[classified_class] + 1
                else:
                    classified_class_count[classified_class] = 1
            continue
    if classified_class_count:
        MaxValue = max(classified_class_count.items(), key=lambda x: x[1]) 
        listclassification = list()
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in classified_class_count.items():
            if value == MaxValue[1]:
                listclassification.append(key)
        if len(listclassification)==4:
            max_Classified="Happy-Active Class"  
        elif len(listclassification)==2 or len(listclassification)==3:
            res = [i for i in listclassification if "Unhappy" in i]
            if(len(res)>(len(listclassification)-len(res))):
                max_Classified="Unhappy-Active Class"               
            elif((len(listclassification)-len(res))>len(res)):
                max_Classified="Happy-Inactive Class"
            elif((len(listclassification)-len(res))==len(res)):
                max_Classified="Happy-Active Class"               
        else:
            max_Classified=listclassification
    else:
        max_Classified="neutral"
    return max_Classified


# In[45]:


data_read['classified']=data_read['post_posConversion'].apply(lambda s:ruleBasedClassification(s))


# In[46]:


data_read['classified']


# In[47]:


X = data_read['Tweets']
y = data_read['Polarity']
test = data_test['Tweets']


# In[48]:


train_tweets = data_read[['Polarity','Tweets']]
test = data_test['Tweets']


# In[49]:


def text_processing(tweet):
    
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    
    return normalization(no_punc_tweet)


# In[58]:


#Machine Learning Pipeline
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  
])
pipeline.fit(msg_train,label_train)


# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from textblob import TextBlob
from nltk.corpus import stopwords
msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['Tweets'], train_tweets['Polarity'], test_size=0.2)


# In[53]:


predictions = pipeline.predict(msg_test)
print("KFold CrossValidation Results")
print(classification_report(predictions,label_test))
print ('\n')
print("Overall Accuracy:")
print(accuracy_score(predictions,label_test))


# In[ ]:




