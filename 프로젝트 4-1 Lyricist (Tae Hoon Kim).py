#!/usr/bin/env python
# coding: utf-8

# In[2]:


#데이터있는 자료를 받는다 for Cloudshell
mkdir -p /aiffel/lyricist/models
ln -s ~/data ~/aiffel/lyricist/data


# In[3]:


#데이터를 읿느다
import os
import re
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split


txt_file_path = os.getenv('HOME')+'/aiffel/lyricist/data/lyrics/*'

txt_list = glob.glob(txt_file_path)

raw_corpus = []

# 여러개의 txt 파일을 모두 읽어서 raw_corpus 에 담습니다.
for txt_file in txt_list:
    with open(txt_file, "r") as f:
        raw = f.read().splitlines()
        raw_corpus.extend(raw)

print("데이터 크기:", len(raw_corpus))
print("Examples:\n", raw_corpus[:3])


# In[5]:


def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()                   
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)    
    sentence = re.sub(r'[" "]+', " ", sentence)           
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)  
    sentence = sentence.strip()                          
    sentence = '<start> ' + sentence + ' <end>'
    return sentence


# In[6]:


corpus = []  
for sentence in raw_corpus:
    if len(sentence) == 0: continue
    tmp = preprocess_sentence(sentence)
    if len(tmp.split()) > 15: continue
    corpus.append(tmp)


# In[7]:


def tokenize(corpus):
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=12000, filters=' ', oov_token="<unk>")
    tokenizer.fit_on_texts(corpus)  # corpus로부터 Tokenizer가 사전을 자동구축

    # tokenizer를 활용하여 모델에 입력할 데이터셋 구축(Tensor로 변환)
    tensor = tokenizer.texts_to_sequences(corpus)

    # 입력 데이터 시퀀스 길이 맞춰주기 - padding
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=15)

    return tensor, tokenizer

tensor, tokenizer = tokenize(corpus)


# In[8]:


for idx in tokenizer.index_word:
    print(idx, ":", tokenizer.index_word[idx])
    if idx >= 5: break

src_input = tensor[:, :-1]
tgt_input = tensor[:, 1:]


# In[9]:


enc_train, enc_val, dec_train, dec_val = train_test_split(src_input, tgt_input, test_size=0.2, random_state=20)
print("Source Train:", enc_train.shape)
print("Target Train:", dec_train.shape)


# In[10]:


class TextGenerator(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(TextGenerator, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)  
        self.rnn_1 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.rnn_2 = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.linear = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        out = self.embedding(x)
        out = self.rnn_1(out)
        out = self.rnn_2(out)
        out = self.linear(out)

        return out


embedding_size = 256
hidden_size = 3000
model = TextGenerator(tokenizer.num_words + 1, embedding_size, hidden_size)


# In[ ]:


optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
model.compile(loss=loss, optimizer=optimizer)
model.fit(enc_train, dec_train, epochs=10, validation_data=(enc_val, dec_val))


def generate_text(model, tokenizer, init_sentence="<start>", max_len=20):
    
    test_input = tokenizer.texts_to_sequences([init_sentence])
    test_tensor = tf.convert_to_tensor(test_input, dtype=tf.int64)
    end_token = tokenizer.word_index["<end>"]

    while True:
        predict = model(test_tensor)  
        predict_word = tf.argmax(tf.nn.softmax(predict, axis=-1), axis=-1)[:, -1] 

        
        test_tensor = tf.concat([test_tensor, tf.expand_dims(predict_word, axis=0)], axis=-1)

        
        if predict_word.numpy()[0] == end_token: break
        if test_tensor.shape[1] >= max_len: break

    generated = ""
    
    for word_index in test_tensor[0].numpy():
        generated += tokenizer.index_word[word_index] + " "

    return generated



test_sen = generate_text(model, tokenizer, init_sentence="<start> My", max_len=20)
print(test_sen)


# In[ ]:




