# chatbot.py
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, LayerNormalization
import string
import re
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def chatbot_response(user_input):
    return "You said: " + user_input


df = pd.read_csv('dialogs.txt', sep='\t', names=['question', 'answer'])
print(f'Dataframe size: {len(df)}')
df.head()

"""# Data Preprocessing

## Data Visualization
"""

df['question tokens'] = df['question'].apply(lambda x: len(x.split()))
df['answer tokens'] = df['answer'].apply(lambda x: len(x.split()))
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
sns.set_palette('Set2')
sns.histplot(x=df['question tokens'], data=df, kde=True, ax=ax[0])
sns.histplot(x=df['answer tokens'], data=df, kde=True, ax=ax[1])
sns.jointplot(x='question tokens', y='answer tokens',
              data=df, kind='kde', fill=True, cmap='YlGnBu')
plt.show()

"""## Text Cleaning"""


def clean_text(text):
    text = re.sub('-', ' ', text.lower())
    text = re.sub('[.]', ' . ', text)
    text = re.sub('[1]', ' 1 ', text)
    text = re.sub('[2]', ' 2 ', text)
    text = re.sub('[3]', ' 3 ', text)
    text = re.sub('[4]', ' 4 ', text)
    text = re.sub('[5]', ' 5 ', text)
    text = re.sub('[6]', ' 6 ', text)
    text = re.sub('[7]', ' 7 ', text)
    text = re.sub('[8]', ' 8 ', text)
    text = re.sub('[9]', ' 9 ', text)
    text = re.sub('[0]', ' 0 ', text)
    text = re.sub('[,]', ' , ', text)
    text = re.sub('[?]', ' ? ', text)
    text = re.sub('[!]', ' ! ', text)
    text = re.sub('[$]', ' $ ', text)
    text = re.sub('[&]', ' & ', text)
    text = re.sub('[/]', ' / ', text)
    text = re.sub('[:]', ' : ', text)
    text = re.sub('[;]', ' ; ', text)
    text = re.sub('[*]', ' * ', text)
    text = re.sub('[\']', ' \' ', text)
    text = re.sub('[\"]', ' \" ', text)
    text = re.sub('\t', ' ', text)
    return text


df.drop(columns=['answer tokens', 'question tokens'], axis=1, inplace=True)
df['encoder_inputs'] = df['question'].apply(clean_text)
df['decoder_targets'] = df['answer'].apply(clean_text)+' <end>'
df['decoder_inputs'] = '<start> '+df['answer'].apply(clean_text)+' <end>'

df.head(10)

df['encoder input tokens'] = df['encoder_inputs'].apply(
    lambda x: len(x.split()))
df['decoder input tokens'] = df['decoder_inputs'].apply(
    lambda x: len(x.split()))
df['decoder target tokens'] = df['decoder_targets'].apply(
    lambda x: len(x.split()))
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
sns.set_palette('Set2')
sns.histplot(x=df['encoder input tokens'], data=df, kde=True, ax=ax[0])
sns.histplot(x=df['decoder input tokens'], data=df, kde=True, ax=ax[1])
sns.histplot(x=df['decoder target tokens'], data=df, kde=True, ax=ax[2])
sns.jointplot(x='encoder input tokens', y='decoder target tokens',
              data=df, kind='kde', fill=True, cmap='YlGnBu')
plt.show()

print(
    f"After preprocessing: {' '.join(df[df['encoder input tokens'].max()==df['encoder input tokens']]['encoder_inputs'].values.tolist())}")
print(f"Max encoder input length: {df['encoder input tokens'].max()}")
print(f"Max decoder input length: {df['decoder input tokens'].max()}")
print(f"Max decoder target length: {df['decoder target tokens'].max()}")

df.drop(columns=['question', 'answer', 'encoder input tokens',
        'decoder input tokens', 'decoder target tokens'], axis=1, inplace=True)
params = {
    "vocab_size": 2500,
    "max_sequence_length": 30,
    "learning_rate": 0.008,
    "batch_size": 149,
    "lstm_cells": 256,
    "embedding_dim": 256,
    "buffer_size": 10000
}
learning_rate = params['learning_rate']
batch_size = params['batch_size']
embedding_dim = params['embedding_dim']
lstm_cells = params['lstm_cells']
vocab_size = params['vocab_size']
buffer_size = params['buffer_size']
max_sequence_length = params['max_sequence_length']
df.head(10)

"""## Tokenization"""

vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    standardize=None,
    output_mode='int',
    output_sequence_length=max_sequence_length
)
vectorize_layer.adapt(df['encoder_inputs']+' ' +
                      df['decoder_targets']+' <start> <end>')
vocab_size = len(vectorize_layer.get_vocabulary())
print(f'Vocab size: {len(vectorize_layer.get_vocabulary())}')
print(f'{vectorize_layer.get_vocabulary()[:12]}')


def sequences2ids(sequence):
    return vectorize_layer(sequence)


def ids2sequences(ids):
    decode = ''
    if type(ids) == int:
        ids = [ids]
    for id in ids:
        decode += vectorize_layer.get_vocabulary()[id]+' '
    return decode


x = sequences2ids(df['encoder_inputs'])
yd = sequences2ids(df['decoder_inputs'])
y = sequences2ids(df['decoder_targets'])

print(f'Question sentence: hi , how are you ?')
print(f'Question to tokens: {sequences2ids("hi , how are you ?")[:10]}')
print(f'Encoder input shape: {x.shape}')
print(f'Decoder input shape: {yd.shape}')
print(f'Decoder target shape: {y.shape}')

print(f'Encoder input: {x[0][:12]} ...')
# shifted by one time step of the target as input to decoder is the output of the previous timestep
print(f'Decoder input: {yd[0][:12]} ...')
print(f'Decoder target: {y[0][:12]} ...')

data = tf.data.Dataset.from_tensor_slices((x, yd, y))
data = data.shuffle(buffer_size)

train_data = data.take(int(.9*len(data)))
train_data = train_data.cache()
train_data = train_data.shuffle(buffer_size)
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(tf.data.AUTOTUNE)
train_data_iterator = train_data.as_numpy_iterator()

val_data = data.skip(int(.9*len(data))).take(int(.1*len(data)))
val_data = val_data.batch(batch_size)
val_data = val_data.prefetch(tf.data.AUTOTUNE)

_ = train_data_iterator.next()
print(f'Number of train batches: {len(train_data)}')
print(f'Number of training data: {len(train_data)*batch_size}')
print(f'Number of validation batches: {len(val_data)}')
print(f'Number of validation data: {len(val_data)*batch_size}')
print(f'Encoder Input shape (with batches): {_[0].shape}')
print(f'Decoder Input shape (with batches): {_[1].shape}')
print(f'Target Output shape (with batches): {_[2].shape}')
