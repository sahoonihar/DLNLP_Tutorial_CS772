# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, FastTex
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from string import punctuation
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))
stop_words.remove("very")
stop_words.add("th")
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

print(torch.__version__)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

train_file = "./train.csv"
test_file = "./test.csv"
gold_file = "./gold_test.csv"

def load_glove(word_index, file):
    
    max_features = len(word_index)+1
    
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file,'r', encoding="utf-8") if  o.split(" ")[0] in word_index)
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    count = 0
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    #     else:
    #         print ("check", word)
    #         count += 1
    # print ("Number of words not available in embedding :",count)
      
    return embedding_matrix  # weight-matrix

def load_trained_embed(word_index, embedding):
    
    max_features = len(word_index)+1
    
    def get_coefs(word,embedding): 
        return word, np.asarray(embedding[word], dtype='float32')

    embeddings_index = dict(get_coefs(o, embedding) for o in embedding.wv.vocab.keys() if  o in word_index)
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    count = 0
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: 
            embedding_matrix[i] = embedding_vector
    #     else:
    #         print ("check", word)
    #         count += 1
    # print ("Number of words not available in embedding :",count)
      
    return embedding_matrix  # weight-matrix

def encode_data(tokenizer, text):
    
    word_index = tokenizer.word_index  # word-to-idx
    print ("Number of tokens : ",len(word_index))
    
    X = tokenizer.texts_to_sequences(text)
    
    return X, word_index

def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    l = []
    for t in text:
        l.append(t.lower())
    return l

def remove_punctuation(text):
    # return the reviews after removing punctuations
    l = []
    for t in text:
        l.append(re.sub(r'[^\w\s]|^\s\d+\s|\s\d+|\d+|\s\d+$', ' ', t)) 
    return l

def remove_stopwords(text):
    # return the reviews after removing the stopwords
    l = []
    large = 0
    for t in text:
        word_tokens = word_tokenize(t)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        l.append(filtered_sentence)
        # large = max(large, len(filtered_sentence))
    # print ("highest :", large)
    return l

def perform_tokenization(text):
    # return the reviews after performing tokenization
    max_features = None
    tokenizer = Tokenizer(num_words=max_features, filters='', oov_token='<OOV>')
    tokenizer.fit_on_texts(list(text))

    return tokenizer

def perform_padding(data):
    # return the reviews after padding the reviews to maximum length
    maxlen = 30
    return pad_sequences(data, maxlen=maxlen, padding="post")

def preprocess_data(data, embedding=None, train=False, tokenizer=None):    
    
    glove = './glove.840B.300d.txt'
    word2vec =  './GoogleNews-vectors-negative300.bin.gz'
    fasttext = './wiki-news-300d-1M.vec'
    
    reviews = convert_to_lower(data)
    reviews = remove_punctuation(reviews)
    reviews = remove_stopwords(reviews)
    if (train):
        tokenizer = perform_tokenization(reviews)
        reviews, word_index = encode_data(tokenizer, reviews)
    else:
        reviews, word_index = encode_data(tokenizer, reviews)
    reviews = perform_padding(reviews)
    if (train):
        if embedding==None:
            # word2vec_embed = load_word2vec(word_index, word2vec)
            # return reviews, word2vec_embed, tokenizer
            glove_embed = load_glove(word_index, glove)
            return reviews, glove_embed, tokenizer
            #fasttext_embed = load_fasttext(word_index, fasttext)
            #return reviews, fasttext_embed, tokenizer
        else:
            glove_embed = load_trained_embed(word_index, embedding)
            return reviews, glove_embed, tokenizer
        
    else:
        return reviews
        
    
def softmax_activation(x):
    # write your own implementation from scratch and return softmax values(using predefined softmax is prohibited)
    
    x = x - torch.max(x, 1)[0][:, None]
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=1, keepdim=True)
    return exp_x/sum_exp_x


class NeuralNet(nn.Module):

    def __init__(self, weights, n_features, hidden, n_layers, n_classes):
        super(NeuralNet, self).__init__()
        
        self.hidden_dim = hidden
        self.n_layers = n_layers
        
        num_embeddings, embedding_dim = weights.shape
        # embedding layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.lstm = torch.nn.LSTM(input_size = embedding_dim, 
                              hidden_size= hidden, 
                              num_layers = n_layers,
                              dropout=0.5,
                              batch_first=True)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc = nn.Linear(hidden, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        
        batch_size = x.size(0)
        
        embed = self.embedding(x)

        # Initialization for hidden states
#         torch.nn.init.xavier_normal_(h)
        lstm_out, hidden_state = self.lstm(embed)
#         lstm_out = torch.cat((hidden_state[-2,:,:],hidden_state[-1,:,:]),dim=1)
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        lstm_out = torch.cat((avg_pool, max_pool), 1)
        hidden = self.relu(self.fc1(lstm_out))

        out = self.dropout(hidden)
        out = self.fc(hidden)
        return out
    
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                 weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        
        return hidden
    

def train_nn(model, EPOCHS, train_loader, val_loader, val_y, BATCH_SIZE, n_classes, class_weights):
    
    criterion = nn.CrossEntropyLoss(reduction='sum')#weight=class_weights
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    train_loss = []
    val_loss = []
    clip = 1.0
    
    
    for epoch in range(EPOCHS):
        
        start_time = time.time()
        # training
        counter = 0
        avg_train_loss = []
        
        for i, (x_batch, y_batch) in enumerate(train_loader):
            
            counter += 1
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            avg_train_loss.append(loss.item() / len(train_loader))
            
        print("Epoch : {}/{}...".format(epoch+1, EPOCHS),
              "Train Loss: {:.4f}...".format(np.sum(avg_train_loss)),
              "Time: {}".format(time.time() - start_time))
        
        val_loss_epoch, val_preds = predict_nn(model, epoch, EPOCHS, val_loader, val_y, BATCH_SIZE, n_classes, criterion)
        train_loss.append(np.sum(avg_train_loss))
        val_loss.append(np.sum(val_loss_epoch))
        scheduler.step()
    
    return model, train_loss, val_loss, val_preds
                
def predict_nn(model, epoch, EPOCHS, val_loader, val_y, BATCH_SIZE, n_classes, criterion):
    
    start_time = time.time()
    # validation
    val_probs = []
    avg_val_loss = []
    val_preds = np.zeros((len(val_y),n_classes))
    
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(val_loader):

            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            y_pred = model(x_batch)
            y_pred = y_pred.detach()
            loss = criterion(y_pred, y_batch)
            avg_val_loss.append(loss.item() / len(val_loader))

            # store predictions
            val_preds[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = y_pred.cpu().numpy() #F.softmax(y_pred).cpu().numpy()

    # check accuracy
    val_prediction = val_preds.argmax(axis=1)
    val_acc = sum(val_prediction == val_y) / len(val_y)

    print("Epoch : {}/{}...".format(epoch+1, EPOCHS),
          "Val Acc: {}...".format(val_acc),
          "Val Loss: {:.4f}...".format(np.sum(avg_val_loss)),
          "Time: {}".format(time.time() - start_time))
    print ("================================================================================")
    if epoch == EPOCHS-1:
        print(classification_report(val_y, val_prediction, target_names = ['1', '2', '3', '4', '5']))
        print (confusion_matrix(val_y, val_prediction))
    return avg_val_loss, val_preds

def predict(review, tokenizer, model):
    review = [review]
    processed_review = preprocess_data(review, None, False, tokenizer)
    X = torch.tensor(processed_review, dtype = torch.long).to(DEVICE)
#     h = tuple([each.data for each in h])
    with torch.no_grad():
        Y = model(X).detach()
        probabilities = softmax_activation(Y)
        label = probabilities.cpu().numpy().argmax()
    return probabilities.cpu().numpy(), label+1

def main(train_file, gold_file, test_file):
    
    BATCH_SIZE,EPOCHS= 32,15

    train_data, gold_data, test_data = pd.read_csv(train_file, delimiter = ','), pd.read_csv(gold_file, delimiter = ','), pd.read_csv(test_file)
    train, train_ratings_li = train_data['reviews'].to_list(), train_data['ratings'].to_list()
    gold, gold_ratings_li = gold_data['reviews'].to_list(), gold_data['ratings'].to_list()
    
    
    train_ratings = np.asarray([r-1 for r in train_ratings_li]) # 0-index the labels
    gold_ratings = np.asarray([r-1 for r in gold_ratings_li]) # 0-index the labels
    val_y = gold_ratings
    
    classes = set(train_ratings)
    n_classes = len(classes)
    
    class1, class2, class3, class4, class5 = np.bincount(train_ratings)  # neg = not default, pos = default
    total = class1 + class2 + class3 + class4 + class5
    print('Examples:\n    Total: {}\n    class1: {} ({:.2f}% of total)\n  class2: {} ({:.2f}% of total)\n   class3: {} ({:.2f}% of total)\n   class4: {} ({:.2f}% of total)\n   class5: {} ({:.2f}% of total)\n'.format(
        total, class1, 100 * class1 / total, class2, 100 * class2 / total, class3, 100 * class3 / total, class4, 100 * class4 / total, class5, 100 * class5 / total))
    
    weight_for_1 = (1 / class1)*(total)
    weight_for_2 = (1 / class2)*(total)
    weight_for_3 = (1 / class3)*(total)
    weight_for_4 = (1 / class4)*(total)
    weight_for_5 = (1 / class5)*(total)
    
    class_weight = {0: weight_for_1, 1: weight_for_2, 2: weight_for_3, 3: weight_for_4, 4: weight_for_5}
    
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    print('Weight for class 2: {:.2f}'.format(weight_for_2))
    print('Weight for class 3: {:.2f}'.format(weight_for_3))
    print('Weight for class 4: {:.2f}'.format(weight_for_4))
    print('Weight for class 5: {:.2f}'.format(weight_for_5))
    
    
    train_data, weight_matrix, tokenizer = preprocess_data(train, embedding=None, train=True)
    gold_data = preprocess_data(gold, embedding=None, train=False, tokenizer=tokenizer)
    
    
    class_weight_list = list(class_weight.values())
    class_weights = torch.FloatTensor(class_weight_list)
    sample_weights = [0] * train_data.shape[0]
    
    for i in range(train_data.shape[0]):
        weight = class_weight_list[train_ratings[i]]
        sample_weights[i] = weight
    
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))
    
    # creates Tensor datasets
    train_data_tensor = torch.tensor(train_data, dtype=torch.long)
    train_ratings_tensor = torch.tensor(train_ratings, dtype=torch.long)
    gold_data_tensor = torch.tensor(gold_data, dtype=torch.long)
    gold_ratings_tensor = torch.tensor(gold_ratings, dtype=torch.long)

    train_set = TensorDataset(train_data_tensor, train_ratings_tensor)
    val_set = TensorDataset(gold_data_tensor, gold_ratings_tensor)
    
    train_loader = DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=BATCH_SIZE)
        
    n_features = train_data.shape[1]
    hidden, hidden_1, hidden_2 = 128, 1000, 100
    layers = 2
    model = NeuralNet(weight_matrix, n_features, hidden, layers, n_classes).to(DEVICE)  
    
    model, train_loss, val_loss, val_preds = train_nn(model, EPOCHS, train_loader, val_loader, val_y, BATCH_SIZE, n_classes, class_weights)
        
    def plot_graph(epochs, train_loss, val_loss):
        fig = plt.figure(figsize=(12,12))
        plt.title("Train/Validation Loss")
        plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')
        plt.plot(list(np.arange(epochs) + 1), val_loss, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        plt.legend(loc='best')
        plt.show()
        
    plot_graph(EPOCHS, train_loss, val_loss)
    
    text = "good way to keep yourself practicing the smiling. will use it and see ."
    a, b = predict(text, tokenizer, model)
    print ("Predicted class :", b)
    
    return model, tokenizer, val_preds

model, tokenizer, test_preds = main(train_file, gold_file, test_file)



# UI for Sentiment prediction

import tkinter as tk
import tkinter.messagebox 

def show_entry_fields():
    print("Input: %s\n" % (e1.get()))
    prob , label = predict(e1.get(), tokenizer, model)
    prob = ['%.4f' % elem for elem in prob[0]]
    tkinter.messagebox.showinfo("Output",  "Class Probabilities: [%s]\nPredicted Label:%d" %(' , '.join(map(str,prob)), label))

master = tk.Tk()
master.title("Enter the input string...") 
master.geometry('250x100')
tk.Label(master, text="Input:").grid(row=0)
e1 = tk.Entry(master)
e1.grid(row=0, column=2)
tk.Button(master, text='Quit', command=master.destroy).grid(row=5, column=2, sticky=tk.W, pady=4)
tk.Button(master, text='Show', command=show_entry_fields).grid(row=5, column=3, sticky=tk.W, pady=4)
tk.mainloop()

