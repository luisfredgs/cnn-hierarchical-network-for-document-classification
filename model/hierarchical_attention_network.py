import datetime, pickle, os, codecs, re, string
import json
import numpy as np
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.utils import CustomObjectScope
from keras import optimizers
from typing import List, Tuple
from tqdm import tqdm
import string
import gensim, logging
import matplotlib.pyplot as plt
import tensorflow as tf

from model.arc.attention import Attention
from model.arc.kmax_pooling import KMaxPooling
from model.arc.tcn import TCN

np.random.seed(1024)

word_embedding_type = "from_scratch" #@param ["from_scratch", "pre_trained"]

word_vector_model = "fasttext" #@param ["fasttext"]
rnn_type = "GRU" #@param ["LSTM", "GRU"]
learning_rate = 0.001


cnn_type = "kim2014" #@param ["tcn", "kim2014", "k-max"]

def load_subword_embedding_300d(word_index):
    print('load_subword_embedding...')
    embeddings_index = {}
    f = codecs.open("../input/wiki-news-300d-1M-subword.vec", encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    #embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)    
    return embedding_matrix


class HANetwork():
    def __init__(self):
        self.model = None
        self.MAX_SENTENCE_LENGTH = 0
        self.MAX_SENTENCE_COUNT = 0
        self.VOCABULARY_SIZE = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.tokenizer = None
        self.class_count = 2

    def _build_model(self, n_classes=2, embedding_dim=200, embeddings_path=False):
        
        # regularização
        l2_reg = regularizers.l2(0.001)
               
        #embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))
        
        if embeddings_path is not None:

            if word_embedding_type is 'from_scratch':
                # FastText
                #filename = './fasttext_model.txt'                
                filename = 'input/yelp/word_embeddings/fasttext_model.txt'
                model =  gensim.models.FastText.load(filename)

                embeddings_index = model.wv                    
                embedding_matrix = np.zeros( ( len(self.tokenizer.word_index) + 1, embedding_dim) )
                for word, i in self.tokenizer.word_index.items():
                    try:
                        embedding_vector = embeddings_index[word]
                        if embedding_vector is not None:
                            embedding_matrix[i] = embedding_vector
                    except Exception as e:
                        continue

                #print(model.wv.most_similar_cosmul(positive=['great', 'nice'], negative=['bad'], topn=3))               
                del embeddings_index

            else:
                # Fasttext pré-treinado
                embedding_dim = 300
                embedding_matrix = load_subword_embedding_300d(self.tokenizer.word_index)

            
            #embedding_weights = embedding_matrix

        sentence_in = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype='int32', name="input_1")
        
        embedding_trainable = True
                
        if word_embedding_type is 'pre_trained':
            embedding_trainable = False
        
        embedded_word_seq = Embedding(
            self.VOCABULARY_SIZE,
            embedding_dim,
            #weights=[embedding_weights],
            weights=[embedding_matrix],
            input_length=self.MAX_SENTENCE_LENGTH,
            trainable=embedding_trainable,
            #mask_zero=True,
            mask_zero=False,
            name='word_embeddings',)(sentence_in) 
        
        
        
        if cnn_type == "kim2014":
            """
            Conv1D - Aplicação de múltiplos filtros de diferentes dimensões, resultando em múltiplos feature maps.
            Baseado no paper de Yoon et al (2014) - https://arxiv.org/pdf/1408.5882.pdf                        
            """
            dropout = Dropout(0.2)(embedded_word_seq)
            filter_sizes = [3,4,5]
            convs = []
            for filter_size in filter_sizes:
                conv = Conv1D(filters=64, kernel_size=filter_size, padding='same', activation='relu')(dropout)
                pool = MaxPool1D(filter_size)(conv)
                convs.append(pool)

            cnn = Concatenate(axis=1)(convs)
            
        elif cnn_type == "tcn":
            # TCN   
            cnn = TCN(
                nb_filters=64, 
                kernel_size=5, 
                nb_stacks=2, 
                dilations=None, 
                #activation='norm_relu',
                activation='wavenet',
                use_skip_connections=True, 
                dropout_rate=0.1, 
                return_sequences=True, 
                name="TCN_1")(embedded_word_seq)
            
        elif cnn_type == "k-max":
             # CNN + Batch Normalization        
            filter_sizes = [3,4,5]
            convs = []
            for filter_size in filter_sizes:                        
                #sdropout = SpatialDropout1D(0.2)(embedded_word_seq)
                conv = Conv1D(
                        filters=64,
                        kernel_size=filter_size, 
                        padding="same", 
                        activation='relu'
                        #kernel_regularizer=l2_reg
                    )(embedded_word_seq)                
                batch_normalization = BatchNormalization()(conv)
                relu = Activation("relu")(batch_normalization)            
                pool = KMaxPooling(k=5, axis=1)(relu) # https://arxiv.org/abs/1404.2188    
                #pool = MaxPool1D(filter_size)(relu)
                convs.append(pool)

            cnn = Concatenate(axis=1)(convs)
                      
        
        if rnn_type is 'GRU':
            word_encoder = Bidirectional(GRU(50, return_sequences=True, dropout=0.2))(cnn)
            
        else:
            word_encoder = Bidirectional(
                LSTM(50, return_sequences=True, dropout=0.2))(embedded_word_seq)
            
        
        dense_transform_w = Dense(
            100, 
            activation='relu', 
            name='dense_transform_w', 
            kernel_regularizer=l2_reg)(word_encoder)
        
        # word attention
        attention_weighted_sentence = Model(
            sentence_in, Attention(name="word_attention")(dense_transform_w))
        
        self.word_attention_model = attention_weighted_sentence
        
        attention_weighted_sentence.summary()

        # sentence-attention-weighted document scores
        
        texts_in = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype='int32', name="input_2")
        
        attention_weighted_sentences = TimeDistributed(attention_weighted_sentence)(texts_in)
        
        
        if rnn_type is 'GRU':
            
            #sentence_encoder = Bidirectional(GRU(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.2))(attention_weighted_sentences)                  
            
            sentence_encoder = Bidirectional(CuDNNGRU(
            units=50, kernel_regularizer=keras.regularizers.l2(1e-5), 
            bias_regularizer=keras.regularizers.l1(1e-3), return_sequences=True))(attention_weighted_sentences)
                
        else:
            sentence_encoder = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.2))(attention_weighted_sentences)
        
        
        dense_transform_s = Dense(
            100, 
            activation='relu', 
            name='dense_transform_s',
            kernel_regularizer=l2_reg)(sentence_encoder)
        
        # sentence attention
        attention_weighted_text = Attention(name="sentence_attention")(dense_transform_s)
        
        
        prediction = Dense(n_classes, activation='softmax')(attention_weighted_text)
        
        model = Model(texts_in, prediction)
        model.summary()
        
        optimizer=Adam(lr=learning_rate, decay=0.0001)

        model.compile(
                      optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def load_weights(self, saved_model_dir, saved_model_filename):
        with CustomObjectScope({'Attention': Attention}):
            print(os.path.join(saved_model_dir, saved_model_filename))
            self.model = load_model(os.path.join(saved_model_dir, saved_model_filename))            
            self.word_attention_model = self.model.get_layer('time_distributed_1').layer
            tokenizer_path = os.path.join(
                saved_model_dir, self._get_tokenizer_filename(saved_model_filename))
            tokenizer_state = pickle.load(open(tokenizer_path, "rb" ))
            self.tokenizer = tokenizer_state['tokenizer']
            self.MAX_SENTENCE_COUNT = tokenizer_state['maxSentenceCount']
            self.MAX_SENTENCE_LENGTH = tokenizer_state['maxSentenceLength']
            self.VOCABULARY_SIZE = tokenizer_state['vocabularySize']
            self._create_reverse_word_index()

    def _get_tokenizer_filename(self, saved_model_filename):
        return saved_model_filename + '.tokenizer'

    def _fit_on_texts(self, texts):
        self.tokenizer = Tokenizer(filters='"()*,-/;[\]^_`{|}~', oov_token='UNK');
        all_sentences = []
        max_sentence_count = 0
        max_sentence_length = 0
        for text in texts:
            sentence_count = len(text)
            if sentence_count > max_sentence_count:
                max_sentence_count = sentence_count
            for sentence in text:
                sentence_length = len(sentence)
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length
                all_sentences.append(sentence)

        self.MAX_SENTENCE_COUNT = min(max_sentence_count, 15)
        self.MAX_SENTENCE_LENGTH = min(max_sentence_length, 50)
        
        self.tokenizer.fit_on_texts(all_sentences)        
        
        del all_sentences
        
        self.VOCABULARY_SIZE = len(self.tokenizer.word_index) + 1
        self._create_reverse_word_index()

    def _create_reverse_word_index(self):
        self.reverse_word_index = {value:key for key,value in self.tokenizer.word_index.items()}

    def _encode_texts(self, texts):
        encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH))
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text), 
                maxlen=self.MAX_SENTENCE_LENGTH))[:self.MAX_SENTENCE_COUNT]
            encoded_texts[i][-len(encoded_text):] = encoded_text
        return encoded_texts

    def _save_tokenizer_on_epoch_end(self, path, epoch):
        if epoch == 0:
            tokenizer_state = {
                'tokenizer': self.tokenizer,
                'maxSentenceCount': self.MAX_SENTENCE_COUNT,
                'maxSentenceLength': self.MAX_SENTENCE_LENGTH,
                'vocabularySize': self.VOCABULARY_SIZE
            }
            pickle.dump(tokenizer_state, open(path, "wb" ) )

    def train(self, 
              train_x, 
              train_y,              
              batch_size=16, 
              epochs=1, 
              embedding_dim=200, 
              embeddings_path=False, 
              saved_model_dir='saved_models', 
              saved_model_filename=None):
        
        # fit tokenizer
        self._fit_on_texts(train_x)
        
        self.model = self._build_model(
            n_classes=train_y.shape[-1], 
            embedding_dim=200,
            embeddings_path=embeddings_path)
        
        encoded_train_x = self._encode_texts(train_x)        
        
        callbacks = [
            EarlyStopping(
               monitor='val_acc',
               patience=3,
             ),
            ReduceLROnPlateau(),            
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self._save_tokenizer_on_epoch_end(
                    os.path.join(saved_model_dir, 
                        self._get_tokenizer_filename(saved_model_filename)), epoch))
        ]

        if saved_model_filename:
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(saved_model_dir, saved_model_filename),
                    monitor='val_acc',
                    save_best_only=True,
                    save_weights_only=False,
                )
            )
        
        history = self.model.fit(
                       x=encoded_train_x, 
                       y=train_y, 
                       batch_size=batch_size, 
                       epochs=epochs, 
                       verbose=1, 
                       callbacks=callbacks,
                       validation_split=0.1,  
                       shuffle=True)
        
        
        # Plot
        print(history.history.keys())
        
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('acc.png')
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('loss.png')

    def _encode_input(self, x, log=False):
        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([normalize(text) for text in x])
        return self._encode_texts(texts)

    def predict(self, x):
        encoded_x = self._encode_texts(x)
        return self.model.predict(encoded_x)
