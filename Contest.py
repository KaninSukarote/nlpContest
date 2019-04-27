import numpy as np
import deepcut
from keras.models import Model
from keras.layers import Input, Dense, GRU, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.metrics import confusion_matrix


#-----------------Read Data-------------------#
#-----------------Read x_train------------------#
file = open('input\input.txt','r',encoding='utf-8')

full_sentences= [s.replace('\n','',) for s in file]
data = [s.split("::") for s in full_sentences]
sentences = [d[1]for d in data]
input_words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in sentences]
sentence_lengths = []
for sentence in input_words:
    sentence_lengths.append(len(sentence))
max_length = max(sentence_lengths)
#-----------------Read y_train------------------#
file = open('result\yTrain.txt','r',encoding='utf-8')
ans= [s.replace('\n','',) for s in file]
ans_data = [s.split("::") for s in ans]
labels = [d[1] for d in ans_data]

labels = np.array(labels)
labels[labels == 'H'] = 0
labels[labels == 'P'] = 1
labels[labels == 'M'] = 2
labels = [(int)(d) for d in labels]
#-----------------Read x_test------------------#
file = open('input\\xTest.txt','r',encoding='utf-8')
full_test= [s.replace('\n','',) for s in file]
test = [s.split("::") for s in full_test]
test_sentences = [d[1]for d in test]
test_words = [[w for w in deepcut.tokenize(s) if w != ' '] for s in test_sentences]
# #-----------------Read y_test------------------#
file = open('result\yTest.txt','r',encoding='utf-8')
full_ans_test= [s.replace('\n','',) for s in file]
ans_test = [s.split("::") for s in full_ans_test]
y_test = [d[1] for d in ans_test]

y_test = np.array(y_test)
y_test[y_test == 'H'] = 0
y_test[y_test == 'P'] = 1
y_test[y_test == 'M'] = 2
y_test = [(int)(d) for d in y_test]
#----------------Extract Word Vector---------------------#
# vocab = set([w for s in words for w in s])

# pretrained_word_vec_file = open('cc.th.300.vec', 'r',encoding = 'utf-8-sig')
# count = 0
# vocab_vec = {}
# for line in pretrained_word_vec_file:
#     if count > 0:
#         line = line.split()
#         if(line[0] in vocab):
#             vocab_vec[line[0]] = line[1:]
#     count = count + 1

# word_vectors = np.zeros((len(words),max_length,100))
# sample_count = 0
# for s in words:
#     word_count = 0
#     for w in s:
#         try:
#             word_vectors[sample_count,19-word_count,:] = vocab_vec[w]
#             word_count = word_count+1
#         except:
#             pass
#     sample_count = sample_count+1

# print(word_vectors.shape)
# print(len(word_vectors[0]))
# print(len(word_vectors[1]))
#-------------------------Bag of Word----------------------------#
words = np.array(input_words+test_words)
print('Word size = '+str(len(words)))
vocab = set([w for s in words for w in s])
print('Vocab size = '+str(len(vocab)))

bag_of_words = np.zeros((len(words),len(vocab)))
for i in range(0,len(words)):
    count = 0
    for j in range(0,len(words[i])):
        k = 0
        for w in vocab:
            if(words[i][j] == w):
                bag_of_words[i][k] = bag_of_words[i][k]+1
                count = count+1
            k = k+1
    bag_of_words[i] = bag_of_words[i]/count


#-----------------Split the data --------------------------------#

x_train, x_valid, y_train, y_valid = train_test_split(bag_of_words[:2363], labels, test_size=0.0, shuffle= True)
#print(x_train)
print('lenght x train = '+str(len(x_train))+'lenght x valid = '+str(len(x_valid)))
#print(x_valid)
print('lenght y train = '+str(len(y_train))+'lenght y valid = '+str(len(y_valid)))
#--------------- Create recurrent neural network-----------------#
# inputLayer = Input(shape=(93,100,))
# rnn = GRU(30, activation='relu')(inputLayer)
# rnn = Dropout(0.5)(rnn)
# outputLayer = Dense(3, activation='softmax')(rnn)
# model = Model(inputs=inputLayer, outputs=outputLayer)

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

#--------------- Create feedforward neural network-----------------
inputLayer = Input(shape=(len(vocab),))
h1 = Dense(128, activation='relu')(inputLayer)
h2 = Dense(64, activation='relu')(h1)
h3 = Dense(16, activation='relu')(h2)
h4 = Dense(8, activation='relu')(h3)
outputLayer = Dense(3, activation='softmax')(h4)
model = Model(inputs=inputLayer, outputs=outputLayer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#----------------------- Train neural network-----------------------#
history = model.fit(x_train, to_categorical(y_train), epochs=200, batch_size=50,validation_split = 0.2)
#-------------------------- Evaluation------------------------------#
y_data = to_categorical(y_test)
y_pred = model.predict(bag_of_words[2363:])
y_t = [np.argmax(i)for i in y_data]
y_p = [np.argmax(i)for i in y_pred]
cm = confusion_matrix(y_t,y_p)

print('Confusion Matrix')
print(cm)
scores = model.evaluate(bag_of_words[2363:], y_data, verbose=1)
print("loss: ", scores[0])
print("Accuracy:", scores[1])
output = []
for y in y_p:
    if(y == 0):
        output.append('H')

    if(y == 1):
        output.append('P')

    if(y == 2):
        output.append('M')



file = open('output.txt','w')
line = 1
for p in output:
    file.writelines('%s::%s\n'%(line,p))
    line = line +1
print(output)
print(line)
 #---------------------------------------------------------------------#
