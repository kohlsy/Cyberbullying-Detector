from django.shortcuts import render
from .forms import InputForm
from .models import View
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import re
import tensorflow as tf

df = pd.read_csv('sentiments.csv')
df.columns = ["label","text"]
x = df['text'].values
y = df['label'].values

x_train, x_test, y_train, y_test = \
 train_test_split(x, y, test_size=0.1, random_state=123)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100)
tokenizer.fit_on_texts(x)
xtrain= tokenizer.texts_to_sequences(x_train)
xtest= tokenizer.texts_to_sequences(x_test)

vocab_size=len(tokenizer.word_index)+1

maxlen=10
xtrain=tf.keras.preprocessing.sequence.pad_sequences(xtrain,padding='post', maxlen=maxlen)
xtest=tf.keras.preprocessing.sequence.pad_sequences(xtest,padding='post', maxlen=maxlen)

embedding_dim=50
model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size+1,
         output_dim=embedding_dim,
         input_length=maxlen))
model.add(tf.keras.layers.LSTM(units=50,return_sequences=True))
model.add(tf.keras.layers.LSTM(units=10))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy",
     metrics=['accuracy'])
# model.summary()
model.fit(xtrain,y_train, epochs=20, batch_size=16, verbose=False)



def predict(word):


    word_Arr = []
    word_Arr.append(word)
    word_final = tokenizer.texts_to_sequences(word_Arr)
    word_final_final = np.asarray(word_final)

    if word_final_final.size == 0:
        word_final_final = np.asarray([[208]])


    # print(word_final_final)
    newArr = np.zeros(shape=(6, 10))
    newArr[0, 0] = word_final_final

    return ((model.predict(newArr)))



def classify(text_input):

    if re.search('\s', text_input):

        array = text_input.split()
        sum = 0
        newArr = np.zeros(shape=(6, 10))

        for g in range(len(array)):
            word_Arr = []
            word_Arr.append(array[g])

            print("Word Array", word_Arr)

            word_final = tokenizer.texts_to_sequences(word_Arr)
            word_final_final = np.asarray(word_final)

            if word_final_final.size == 0:
                word_final_final = np.asarray([[1]])

            print("Word Final Np Array", word_final_final)

            newArr[0, 0] = word_final_final

            sum += float((model.predict(newArr))[0])

        print(sum/len(array))


    else:
        print(predict(text_input)[0])


# Create your views here.


def index(request):

    form = InputForm()

    args = {'form': form}

    if request.method == "POST":

        print("checking")

        form = InputForm(request.POST)

        if form.is_valid():

            print("test1")

            print((str(form.cleaned_data['textInput'])))

    return render(request, 'main_app/UI.html', args)
