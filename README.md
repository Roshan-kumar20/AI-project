# AI-project
News prediction (whether the given news is good or bad) 
Project-13
NEWS REVIEWER
 
Course Code: Int 404
Submitted to: Sukhvir Kaur 
Date of submission: 10/4/20

Name	Registration no	Roll no
G. kiran kumar reddy	11810726	49
Adabala Arya	11802309	50
Pragy kumar vashistha	11803225	51
Roshan Kumar	11802478	52
Contribution to project
Name	Contribution
G. Kiran Kumar	Code+ Abstract+ Introduction
Arya	Code+ introduction+ Objectives
Pragy Kumar	Code+ Objectives+ Result
Roshan Kumar	Code+ conclusion+ future scope
Content
1.	Abstract………………………………………………………… 1
1.1.	Overview of project…………………………………1.1
1.2.	Reality of project……………………………………. 1.2
2.	Introduction…………………………………………………. 2
2.1.	About concepts of project……………………………. 2.1
2.2.	About the use of project………………………………. 2.2
2.3.	The new concepts we learn…………………………… 2.3
3.	Objectives……………………………………………………… 3
3.1.	Code ……………………………………………………… 3.1
3.2.	Libraries in code…………………………………… 3.2
3.3.	Functionality of libraries……………………… 3.3
3.4.	Functionality of code…………………………… 3.4
4.	Methodology…………………………………………………. 4
4.1.	The methodology followed ……………………. 4.1
4.2.	Challenges in the methodology……………… 4.2
5.	Result …………………………………………………………... 5
5.1.	Output…………………………………………………… 5.1
6.	Conclusion……………………………………………………… 6
7.	Future Scope…………………………………………………… 7

Abstract
1.1. Overview of the project: we got the project about prediction that the news is good or bad. We started the project by reading some newspapers and predicting weather the new Is god or bad, we faced problem with the news which is neither good or bad, we had a idea that the project would be done in the style of weather forecasting. But we faced some problem for that we read many codes of weather forecasting and before starting the code we referred many books we just wanted a small code which can do this task so we started knowing about the libraries because in python libraries make the code easier e referred my libraries and came to conclusion that we can make it simple after that we started, in the video we wanted to explain each concept of the project and we started it.
1.2.	Reality of project: The project was  not as simple as weather forecasting as in weather forecasting we just need to analysis the data and give the forecasting report here in this project we need to understand the module of the data and then we can say the news is good or bad the most difficult problem us some news were good for some people and bad for other example raining heavy was good for farmer and worst for fisherman in this way we were facing the reality problem while completing the problem.
Introduction
2.1.	About concepts of project: Introduction
Sentiment analysis is part of the Natural Language Processing (NLP) techniques that consists in extracting emotions related to some raw texts. This is usually used on social media posts and customer reviews in order to automatically understand if some users are positive or negative and why. The goal of this study is to show how sentiment analysis can be performed using python. Here are some of the main libraries we will use:
NLTK: the most famous python module for NLP techniques
Genism: a topic-modelling and vector space modelling toolkit
2.2.	About the use of project: The production of news weather the news is good bad or neutral gives us a strong grip on the news as we need to modular the news and then classify it into different modules and then we can predict whether the news is good or bad by this we can improve our production level of the new or sentence 
 Here we came across many problems as one news is good for some and bad for some in that case, we had to possibility whether we can go on majority base or we can declare it neutral and we went to majority basis 
2.3.	The new concepts we learn: by this we learn many new skill like understanding the code functions of libraries and many more the most important thing I lent was about the new libraries, It is completely possible to use only raw text as input for making predictions. The most important thing is to be able to extract the relevant features from this raw source of data. This kind of data can often come as a good complementary source in data science projects in order to extract more learning features and increase the predictive power of the models








Objectives
Code
!pip install gensim --upgrade
!pip install keras --upgrade
!pip install pandas --upgrade
# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt
%matplotlib inline

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download('stopwords')
Settings
# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"
Read Dataset
Dataset details
target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
ids: The id of the tweet ( 2087)
date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
flag: The query (lyx). If there is no query, then this value is NO_QUERY.
user: the user that tweeted (robotickilldozr)
text: the text of the tweet (Lyx is cool)
dataset_filename = os.listdir("../input")[0]
dataset_path = os.path.join("..","input",dataset_filename)
print("Open file:", dataset_path)
df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)
print("Dataset size:", len(df))
df.head(5)
Map target label to String
0 -> NEGATIVE
2 -> NEUTRAL
4 -> POSITIVE
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
%%time
df.target = df.target.apply(lambda x: decode_sentiment(x))
target_cnt = Counter(df.target)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")
Pre-Process dataset
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)
%%time
df.text = df.text.apply(lambda x: preprocess(x))
Split train and test
df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))
Word2Vec
%%time
documents = [_text.split() for _text in df_train.text] 
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)
%%time
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
w2v_model.most_similar("love")
Tokenize Text
%%time
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train.text)

vocab_size = len(tokenizer.word_index) + 1
print("Total words", vocab_size)
%%time
x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)
Label Encoder
labels = df_train.target.unique().tolist()
labels.append(NEUTRAL)
labels
encoder = LabelEncoder()
encoder.fit(df_train.target.tolist())

y_train = encoder.transform(df_train.target.tolist())
y_test = encoder.transform(df_test.target.tolist())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)
print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)
y_train[:10]
Embedding layer
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)
embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)
Build Model
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()
Compile model
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
Callbacks
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
Train
%%time
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)
Evaluate
%%time
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()
Predict
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE
def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  
predict("I love the music")
predict("I hate the rain")
predict("i don't know what i'm doing")
Confusion Matrix
%%time
y_pred_1d = []
y_test_1d = list(df_test.target)
scores = model.predict(x_test, verbose=1, batch_size=8000)
y_pred_1d = [decode_sentiment(score, include_neutral=False) for score in scores]
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
%%time

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(12,12))
plot_confusion_matrix(cnf_matrix, classes=df_train.target.unique(), title="Confusion matrix")
plt.show()
Classification Report
print(classification_report(y_test_1d, y_pred_1d))
Accuracy Score
accuracy_score(y_test_1d, y_pred_1d)
Save model
model.save(KERAS_MODEL)
w2v_model.save(WORD2VEC_MODEL)
pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)
pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)






Result  Output  

   
 

Methodology
4.1.	The methodology followed: The next thing is the method which we are going to following to complete the project, we had many things to do, many ways to start the project we can focus on the data base or code or report, we started the project by focusing on the algorithm we made for the project and be started with code and then we collet some data for testing after this success then we divided the report content and finally we made the project.
We referred many machine learning basics books to make a short code for our required output
4.2.	Challenges in the methodology: we faced many challenges and we solved, we faced more project in writing the code and the thing which we were not getting is the meaning of the libraries and choosing out the libraries and after reading all we came to an common conclusion  

Conclusion
AI is at the center of a new enterprise to build computational models of intelligence. The main assumption is that intelligence (human or otherwise) can be represented in terms of symbol structures and symbolic operations which can be programmed in a digital computer. There is much debate as to whether such an appropriately programmed computer would be a mind, or would merely simulate one, but AI researchers need not wait for the conclusion to that debate, nor for the hypothetical computer that could model all of human intelligence. Aspects of intelligent behavior, such as solving problems, making inferences, learning, and understanding language, have already been coded as computer programs, and within very limited domains, such as identifying diseases of soybean plants, AI programs can outperform human experts. Now the great challenge of AI is to find ways of representing the commonsense knowledge and experience that enable people to carry out everyday activities such as holding a wide-ranging conversation or finding their way along a busy street. Conventional digital computers may be capable of running such programs, or we may need to develop new machines that can support the complexity of human thought.

Future Scope
This project is of ai and the future scope AI is going to permeate every job sector in the future. It can create new career paths in the field of Machine learning, Data mining, and analysis, AI software development, program management, and testing. The demand for AI certified professionals will grow along with the developments in AI




 
     

