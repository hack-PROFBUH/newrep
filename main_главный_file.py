from __future__ import division

import models
import data
import main

import sys
import codecs

import tensorflow as tf
import numpy as np

#############     Тут код для скачивания и конертирования видео, а также расшифровки в текст
from pytube import YouTube
from moviepy.editor import *
import speech_recognition as sr

###
link = input("sslka: ")
yt = YouTube(link)
ys = yt.streams.get_highest_resolution()
ys.download(filename="my_video.mp4")
###

'''
path = "my_video.mp4"                           # Для образки видео по таймкодам
start_time = 5
end_time = 15
clip = VideoFileClip(path).subclip(start_time, end_time)
clip.write_videofile("my_video_trimmed.mp4")
'''

###
video_path = "my_video.mp4"
video = VideoFileClip(video_path)
audio = video.audio
audio_path = "audio.wav"
audio.write_audiofile(audio_path)
###

###
r = sr.Recognizer()
file_path = "audio.wav"
with sr.AudioFile(file_path) as source:
    audio_text = r.record(source)

text = r.recognize_google(audio_text, language="ru-RU")
'''print(text)'''     # Получившийся текст
#############

MAX_SUBSEQUENCE_LEN = 50000
model_file = r'Model_ru_punctuator_h256_lr0.02.pcl'   # Модель

def to_array(arr, dtype=np.int32):
    
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def restore(text, word_vocabulary, reverse_punctuation_vocabulary, model):
    i = 0
    while True:
        string_to_punct = ''
        subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

        y = predict(to_array(converted_subsequence), model)

        string_to_punct += subsequence[0]

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(tf.reshape(y_t, [-1]))
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) 

        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            string_to_punct += (punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
            if j < step - 1:
                string_to_punct += subsequence[1+j]

        if subsequence[-1] == data.END:
            break

        i += step
    return(string_to_punct)

def predict(x, model):
    return tf.nn.softmax(net(x))

if __name__ == "__main__":

    vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)


    net, _ = models.load(model_file, x)



    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}
    for key, value in reverse_punctuation_vocabulary.items():
        if value == '.PERIOD':
            reverse_punctuation_vocabulary[key] = '.'
        if value == ',COMMA':
            reverse_punctuation_vocabulary[key] = ','
        if value == '?QUESTIONMARK':
            reverse_punctuation_vocabulary[key] = '?'
    
    
    #input_text = input()
    input_text = text        

    if len(input_text) == 0:
        sys.exit("No")

    text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
    pauses = [float(s.replace(data.PAUSE_PREFIX,"").replace(">","")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]

    text_with_punct = restore(text, word_vocabulary, reverse_punctuation_vocabulary, net)
    import nltk.data
    punkt_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
    sentences = punkt_tokenizer.tokenize(text_with_punct)
    sentences = [sent.capitalize() for sent in sentences]
    uppercase_text = ' '.join(sentences)
    #print(uppercase_text)                 # uppercase_text - Конечный текст со всеми заглавными буквами и знаками препинания


###########################################################################################################################################


'''
import yake                                #Это инструмент для поиска ключевых слов на yake (хороший инструмент, но не пригодится)
                                           #Он отлично подходит для различных видео(не лекция)
###
words = uppercase_text.split()
num_words = len(words)
if num_words > 1000 and num_words < 100:
    wwword = num_words // 30
elif num_words < 100:
    wwword = num_words // 10
###
    
kw_extractor = yake.KeywordExtractor(lan="ru", n=1, top=wwword) #подумать мне надо да
keywords = kw_extractor.extract_keywords(uppercase_text)

words = []

#for kw in keywords:
    #print(kw)
for i in keywords:
    words.append(i[0])

my_dict = {}

#text = text.lower()

text_words = uppercase_text.split()'''

#punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''                 #Далее идет расставление слов по порядку
'''
for i in range(len(text_words)):
    for punctuation in punctuations:
        text_words[i] = text_words[i].replace(punctuation, "")
    text_words[i] = text_words[i].strip()
    text_words[i] = text_words[i].replace(" ", "")

for i in range(0, len(text_words)):
    if text_words[i] in words:
        my_dict[text_words[i]] = i + 1

#print(my_dict)
A = []
my_list = list(my_dict.items())

#print(my_list)        
        
for i in my_list:
    A.append(i[0])
#print(len(A))
#print(len(words))
print(' ')
print(' ')
print(' ')
print("Текст норм -", uppercase_text)
print('Ключевые слова списком -', A)
'''
######################################################################

#  Дальше пойдет обработка текста с использованием API Chat GPT (самое лучшее решение по соотношению скорости и качества)

import openai
openai.api_key = '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'

messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
message = 'Раздели этот текст на абзацы: ' + uppercase_text
if message:
	messages.append(
		{"role": "user", "content": message},
	)
	chat = openai.ChatCompletion.create(
		model="gpt-3.5-turbo", messages=messages
	)

reply = chat.choices[0].message.content
answer = reply
messages.clear()
#print(f"ChatGPT: {reply}")
#messages.append({"role": "assistant", "content": reply})

messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
message = 'Выдели из этого текста очень много ключевых предложений: ' + uppercase_text
if message:
	messages.append(
		{"role": "user", "content": message},
	)
	chat = openai.ChatCompletion.create(
		model="gpt-3.5-turbo", messages=messages
	)

reply = chat.choices[0].message.content

messages.clear()

print(f"Заголовок: {yt.title}")
print(' ')
print(' ')
print(' ')
print(' ')
print(f"ChatGPT: {reply}")


parts = answer.split("\n\n")     # Это список со всеми абзацами

all_parts = []

for part in parts:
    all_parts.append(part)











        
