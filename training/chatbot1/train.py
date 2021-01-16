########################## imports ############################################

import json
import pickle

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import nltk
nltk.download('punkt')
nltk.download('wordnet')

########################## imports ############################################


########################## loading ds ############################################

with open('corpus_ds/intents.json') as df:
    data_file = df.read()
    
intents = json.loads(data_file)

print(intents)

########################## loading ds ############################################


words, classes, documents = [], [], []


########################## tokenizing ds ############################################

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        print(w)
        words.extend(w)
        
        # adding documents
        documents.append((w, intent['tag']))
        print(documents)

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        
        print(classes,'\n\n')

########################## tokenizing ds ############################################


ignore_words = ['?', '!', 'a', 'about', 'appreciate', 'are', 'can', 
                'could', 'do', 'for', 'give', 'i', 'is', 'it', 'me', 
                'the', 'to', 'wa', 'whats', 'who', 'ya', 'i']


########################## making words and classes ############################################

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
words.remove('i')
words.remove('wa')
words.remove('what')
words.remove('who')

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes:", classes)
print (len(words), "unique lemmatized words:", words)

with open('model_utils/words.pkl','wb') as w_pkl:
    pickle.dump(words,w_pkl)
with open('model_utils/classes.pkl','wb') as c_pkl:
    pickle.dump(classes,c_pkl)
    
########################## making words and classes ############################################


########################## making training ds ############################################

# initializing training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    
    # initializing bag of words
    bag = []
    
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])


# shuffle our features and turn into np.array
np.random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")

########################## making training ds ############################################


########################## Training model ############################################

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs = 320, batch_size=1, verbose=2)
model.save('model_weights/chatbot_model.h5', hist)
print("model created")

########################## Training model ############################################
