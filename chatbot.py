import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


df = pd.read_csv('/dialogs.txt', sep='\t', names=['question', 'answer'])
"""# Data Preprocessing

## Text Cleaning"""
df['question tokens'] = df['question'].apply(lambda x: len(x.split()))
df['answer tokens'] = df['answer'].apply(lambda x: len(x.split()))


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

df['encoder input tokens'] = df['encoder_inputs'].apply(
    lambda x: len(x.split()))
df['decoder input tokens'] = df['decoder_inputs'].apply(
    lambda x: len(x.split()))
df['decoder target tokens'] = df['decoder_targets'].apply(
    lambda x: len(x.split()))

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

vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    standardize=None,
    output_mode='int',
    output_sequence_length=max_sequence_length
)
vectorize_layer.adapt(df['encoder_inputs']+' ' +
                      df['decoder_targets']+' <start> <end>')
vocab_size = len(vectorize_layer.get_vocabulary())


def sequences2ids(sequence):
    return vectorize_layer(sequence)


def ids2sequences(ids):
    decode = ''
    if type(ids) == int:
        ids = [ids]
    for id in ids:
        decode += vectorize_layer.get_vocabulary()[id]+' '
    return decode


class ChatBot(tf.keras.models.Model):
    def __init__(self, base_encoder, base_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder, self.decoder = self.build_inference_model(
            base_encoder, base_decoder)

    def build_inference_model(self, base_encoder, base_decoder):
        encoder_inputs = tf.keras.Input(shape=(None,))
        x = base_encoder.layers[0](encoder_inputs)
        x = base_encoder.layers[1](x)
        x, encoder_state_h, encoder_state_c = base_encoder.layers[2](x)
        encoder = tf.keras.models.Model(inputs=encoder_inputs, outputs=[
                                        encoder_state_h, encoder_state_c], name='chatbot_encoder')
        lstm_cells = 256
        decoder_input_state_h = tf.keras.Input(shape=(lstm_cells,))
        decoder_input_state_c = tf.keras.Input(shape=(lstm_cells,))
        decoder_inputs = tf.keras.Input(shape=(None,))
        x = base_decoder.layers[0](decoder_inputs)
        x = base_encoder.layers[1](x)
        x, decoder_state_h, decoder_state_c = base_decoder.layers[2](
            x, initial_state=[decoder_input_state_h, decoder_input_state_c])
        decoder_outputs = base_decoder.layers[-1](x)
        decoder = tf.keras.models.Model(
            inputs=[decoder_inputs, [
                decoder_input_state_h, decoder_input_state_c]],
            outputs=[decoder_outputs, [decoder_state_h,
                                       decoder_state_c]], name='chatbot_decoder'
        )
        return encoder, decoder

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

    def softmax(self, z):
        return np.exp(z)/sum(np.exp(z))

    def sample(self, conditional_probability, temperature=0.5):
        conditional_probability = np.asarray(
            conditional_probability).astype("float64")
        conditional_probability = np.log(conditional_probability) / temperature
        reweighted_conditional_probability = self.softmax(
            conditional_probability)
        probas = np.random.multinomial(
            1, reweighted_conditional_probability, 1)
        return np.argmax(probas)

    def postprocess(self, text):
        text = re.sub(' - ', '-', text.lower())
        text = re.sub(' [.] ', '. ', text)
        text = re.sub(' [1] ', '1', text)
        text = re.sub(' [2] ', '2', text)
        text = re.sub(' [3] ', '3', text)
        text = re.sub(' [4] ', '4', text)
        text = re.sub(' [5] ', '5', text)
        text = re.sub(' [6] ', '6', text)
        text = re.sub(' [7] ', '7', text)
        text = re.sub(' [8] ', '8', text)
        text = re.sub(' [9] ', '9', text)
        text = re.sub(' [0] ', '0', text)
        text = re.sub(' [,] ', ', ', text)
        text = re.sub(' [?] ', '? ', text)
        text = re.sub(' [!] ', '! ', text)
        text = re.sub(' [$] ', '$ ', text)
        text = re.sub(' [&] ', '& ', text)
        text = re.sub(' [/] ', '/ ', text)
        text = re.sub(' [:] ', ': ', text)
        text = re.sub(' [;] ', '; ', text)
        text = re.sub(' [*] ', '* ', text)
        text = re.sub(' [\'] ', '\'', text)
        text = re.sub(' [\"] ', '\"', text)
        return text

    def call(self, text, config=None):
        input_seq = self.preprocess(text)
        states = self.encoder(input_seq, training=False)
        target_seq = np.zeros((1, 1))
        target_seq[:, :] = sequences2ids(['<start>']).numpy()[0][0]
        stop_condition = False
        decoded = []
        while not stop_condition:
            decoder_outputs, new_states = self.decoder(
                [target_seq, states], training=False)
#             index=tf.argmax(decoder_outputs[:,-1,:],axis=-1).numpy().item()
            index = self.sample(decoder_outputs[0, 0, :]).item()
            word = ids2sequences([index])
            max_sequence_length = 30
            if word == '<end> ' or len(decoded) >= max_sequence_length:
                stop_condition = True
            else:
                decoded.append(index)
                target_seq = np.zeros((1, 1))
                target_seq[:, :] = index
                states = new_states
        return self.postprocess(ids2sequences(decoded))


# Load the saved model
loaded_model = tf.keras.models.load_model(
    'D:\Testing_Folder\Phase_3\models')

# Create an instance of the ChatBot class
chatbot = ChatBot(loaded_model.get_layer('chatbot_encoder'),
                  loaded_model.get_layer('chatbot_decoder'))


def chatbot_response(text):
    return chatbot.call(text)
# # Now you can use the chatbot to get responses
# user_input = "Hi"  # Replace with the text you want to generate a response for
# response = chatbot.call(user_input)
# print("Chatbot Response:", response)
