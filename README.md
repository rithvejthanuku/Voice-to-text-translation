from google.colab import drive
import zipfile
impor
Voice to text translation.ipynb
Voice to text translation.ipynb_

[ ]
from google.colab import drive
import zipfile
import os

# Mount Google Drive
drive.mount('/content/drive')


[ ]
import os
import librosa

# Path to your dataset folder
dataset_folder = '/content/nonsensitive'

# Function to read audio files from the dataset folder
def read_dataset(folder):
    dataset = []
    labels = []


[ ]
from collections import Counter


# Count the occurrences of each label
label_counts = Counter(labels)

# Print the label counts
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")

Label: nonsensitive, Count: 99
Label: sensitive, Count: 45

[ ]
import os
import librosa
import numpy as np

# Define the function for preprocessing audio files
def preprocess_audio_folder(folder_path, sample_rate=16000, duration=2):
    preprocessed_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            audio_file = os.path.join(folder_path, filename)
            preprocessed_audio = preprocess_audio(audio_file, sample_rate, duration)
            preprocessed_data.append((preprocessed_audio, filename))  # Store preprocessed audio and filename
    return preprocessed_data

# Function to preprocess a single audio file
def preprocess_audio(audio_file, sample_rate=16000, duration=2):
    # Load audio file
    audio_data, _ = librosa.load(audio_file, sr=sample_rate, duration=duration)

    # Trim silence
    audio_data, _ = librosa.effects.trim(audio_data)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)

    # Pad or truncate MFCCs to a fixed length
    max_len = 100
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]

    return mfccs

# Example usage
folder_path = '/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData'
preprocessed_data = preprocess_audio_folder(folder_path)

# Example print the preprocessed data
for audio_data, filename in preprocessed_data:
    print("Filename:", filename)
    print("Preprocessed Data Shape:", audio_data.shape)
    # Use preprocessed data for further processing or model training


[ ]
pip install SpeechRecognition

Collecting SpeechRecognition
  Downloading SpeechRecognition-3.10.4-py2.py3-none-any.whl (32.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 32.8/32.8 MB 42.1 MB/s eta 0:00:00
Requirement already satisfied: requests>=2.26.0 in /usr/local/lib/python3.10/dist-packages (from SpeechRecognition) (2.31.0)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from SpeechRecognition) (4.11.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->SpeechRecognition) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->SpeechRecognition) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->SpeechRecognition) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.26.0->SpeechRecognition) (2024.2.2)
Installing collected packages: SpeechRecognition
Successfully installed SpeechRecognition-3.10.4

[ ]
import speech_recognition as sr


[ ]
import os
import librosa
import speech_recognition as sr

# Path to your dataset folder
dataset_folder = '/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData'

# Function to read audio files from the dataset folder
def read_dataset(folder):
    dataset = []
    labels = []
    recognizer = sr.Recognizer()
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for audio_file in os.listdir(label_folder):
                if audio_file.endswith('.wav'):  # Assuming your audio files are in WAV format
                    file_path = os.path.join(label_folder, audio_file)
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    text = recognize_audio(recognizer, file_path)
                    dataset.append((audio_data, text))
                    labels.append(label)
    return dataset, labels

# Function to recognize speech from an audio file
def recognize_audio(recognizer, file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Unknown"
        except sr.RequestError as e:
            return "Error: {0}".format(e)

# Read the dataset
data, labels = read_dataset(dataset_folder)

# Example: print the number of audio files, their corresponding labels, and recognized text
for i in range(len(data)):
    print("Audio File:", i+1)
    print("Label:", labels[i])
    print("Recognized Text:", data[i][1])
    print("------------")

Audio File: 1
Label: nonsensitive
Recognized Text: so if there is nothing else we need to discuss let's move on to today's agenda
------------
Audio File: 2
Label: nonsensitive
Recognized Text: the meeting was declared closed at 11.30
------------
Audio File: 3
Label: nonsensitive
Recognized Text: hi it's me
------------
Audio File: 4
Label: nonsensitive
Recognized Text: let me just summarize the main points of the last meeting
------------
Audio File: 5
Label: nonsensitive
Recognized Text: thank you Tom
------------
Audio File: 6
Label: nonsensitive
Recognized Text: have you all received a copy of today's agenda
------------
Audio File: 7
Label: nonsensitive
Recognized Text: you should sign up for that seminar next year
------------
Audio File: 8
Label: nonsensitive
Recognized Text: after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements
------------
Audio File: 9
Label: nonsensitive
Recognized Text: before I begin the report I'd like to get some ideas from you all
------------
Audio File: 10
Label: nonsensitive
Recognized Text: I also need to learn how to better mg my workload I always run out of time
------------
Audio File: 11
Label: nonsensitive
Recognized Text: an advertising campaign to focus on their particular needs
------------
Audio File: 12
Label: nonsensitive
Recognized Text: Unknown
------------
Audio File: 13
Label: nonsensitive
Recognized Text: I think rural customers want to feel as important as our customers living in cities
------------
Audio File: 14
Label: nonsensitive
Recognized Text: well let me begin with this PowerPoint presentation Jack presents his report
------------
Audio File: 15
Label: nonsensitive
Recognized Text: how do you feel about rural sales in your sales districts
------------
Audio File: 16
Label: nonsensitive
Recognized Text: excuse me I didn't catch that
------------
Audio File: 17
Label: nonsensitive
Recognized Text: can we fix the next meeting please
------------
Audio File: 18
Label: nonsensitive
Recognized Text: I suggest we break up into groups and discuss the ideas we've seen presented
------------
Audio File: 19
Label: nonsensitive
Recognized Text: we are considering specific data mining procedures to help deepen our understanding
------------
Audio File: 20
Label: nonsensitive
Recognized Text: I'd like to thank Jack for coming to our meeting today
------------
Audio File: 21
Label: nonsensitive
Recognized Text: and I have to mge a staff of 10 people
------------
Audio File: 22
Label: nonsensitive
Recognized Text: a survey will be completed to collect data on spending habits in these areas
------------
Audio File: 23
Label: nonsensitive
Recognized Text: same here I only have seven people reporting to me yet sometimes I feel like pulling my hair out
------------
Audio File: 24
Label: nonsensitive
Recognized Text: Jack is kindly agreed to give us a report on this matter Jack
------------
Audio File: 25
Label: nonsensitive
Recognized Text: in my opinion we have been focusing too much on Urban customers and their needs
------------
Audio File: 26
Label: nonsensitive
Recognized Text: I must admit I never thought about rural sales that way before I have to agree with Alice
------------
Audio File: 27
Label: nonsensitive
Recognized Text: I don't quite follow you what exactly do you mean
------------
Audio File: 28
Label: nonsensitive
Recognized Text: good idea Donald how does Friday in 2 weeks time sound to everyone let's meet at the same time 9:00
------------
Audio File: 29
Label: nonsensitive
Recognized Text: I suggest we go around the table first to get all of your input
------------
Audio File: 30
Label: nonsensitive
Recognized Text: if you don't mind I'd like to skip item one and move on to item two sales Improvement in rural Market areas
------------
Audio File: 31
Label: nonsensitive
Recognized Text: is that okay for everyone excellent
------------
Audio File: 32
Label: nonsensitive
Recognized Text: before we close let me just summarize the main points
------------
Audio File: 33
Label: nonsensitive
Recognized Text: just like any other company we are quite affected by the slowing economy
------------
Audio File: 34
Label: nonsensitive
Recognized Text: they tend to be bored when things slow down and that is not good
------------
Audio File: 35
Label: nonsensitive
Recognized Text: let's get back to our seats so that we can learn how to bring out the best in ourselves as well as our employees
------------
Audio File: 36
Label: nonsensitive
Recognized Text: I know sometime next month I will have to go to Texas on a business trip
------------
Audio File: 37
Label: nonsensitive
Recognized Text: Lynn please come to my office
------------
Audio File: 38
Label: nonsensitive
Recognized Text: very knowledgeable in the subject matter
------------
Audio File: 39
Label: nonsensitive
Recognized Text: Unknown
------------
Audio File: 40
Label: nonsensitive
Recognized Text: besides we are in the process of updating our computer system
------------
Audio File: 41
Label: nonsensitive
Recognized Text: this is a pretty good seminar so far
------------
Audio File: 42
Label: nonsensitive
Recognized Text: by the way what is your company doing
------------
Audio File: 43
Label: nonsensitive
Recognized Text: and we can use this slow period to finish the process
------------
Audio File: 44
Label: nonsensitive
Recognized Text: it seems like the days are getting shorter and shorter
------------
Audio File: 45
Label: nonsensitive
Recognized Text: I just received a revised purchase order from one of our customers
------------
Audio File: 46
Label: nonsensitive
Recognized Text: it is coming next month
------------
Audio File: 47
Label: nonsensitive
Recognized Text: we better not miss any part of it
------------
Audio File: 48
Label: nonsensitive
Recognized Text: I hate those types of days luckily it is not that bad in our company
------------
Audio File: 49
Label: nonsensitive
Recognized Text: I guess I'll break time is over
------------
Audio File: 50
Label: nonsensitive
Recognized Text: Unknown
------------
Audio File: 51
Label: nonsensitive
Recognized Text: business is slow with my company too
------------
Audio File: 52
Label: nonsensitive
Recognized Text: I have some very talented employees and I would like to keep their mind Sharp
------------
Audio File: 53
Label: nonsensitive
Recognized Text: good for you oh 10:30 a.m. already
------------
Audio File: 54
Label: nonsensitive
Recognized Text: let's see whether I will have any free time next month
------------
Audio File: 55
Label: nonsensitive
Recognized Text: let's hope so we need to keep our employees busy
------------
Audio File: 56
Label: nonsensitive
Recognized Text: how is business lately
------------
Audio File: 57
Label: nonsensitive
Recognized Text: good I just received a revised order from its purchasing department
------------
Audio File: 58
Label: nonsensitive
Recognized Text: it will cost us more if we put in a change of order now
------------
Audio File: 59
Label: nonsensitive
Recognized Text: John Miller
------------
Audio File: 60
Label: nonsensitive
Recognized Text: the production department will have to work a lot of overtime this month
------------
Audio File: 61
Label: nonsensitive
Recognized Text: how is a production Department doing
------------
Audio File: 62
Label: nonsensitive
Recognized Text: it does pay to keep up with the customers demand everybody likes our products and services
------------
Audio File: 63
Label: nonsensitive
Recognized Text: did you attend the seminar on leadership in Long Beach last January
------------
Audio File: 64
Label: nonsensitive
Recognized Text: I better give a copy of this new order to our production Department
------------
Audio File: 65
Label: nonsensitive
Recognized Text: the 30th of this month is a major holiday and its Shipping schedule is going to be very tight
------------
Audio File: 66
Label: nonsensitive
Recognized Text: it is running on a very tight schedule we received quite a few others lately
------------
Audio File: 67
Label: nonsensitive
Recognized Text: the one for a lot of 500 Elkwood windows
------------
Audio File: 68
Label: nonsensitive
Recognized Text: no I missed that one who was the speaker
------------
Audio File: 69
Label: nonsensitive
Recognized Text: I think you should put a call into Trucking lines as soon as possible
------------
Audio File: 70
Label: nonsensitive
Recognized Text: it is not due until the 25th of the month now
------------
Audio File: 71
Label: nonsensitive
Recognized Text: yes that is the one did we start production on it yet
------------
Audio File: 72
Label: nonsensitive
Recognized Text: true they might not accommodate our change if they receive our notice too late
------------
Audio File: 73
Label: nonsensitive
Recognized Text: we have already ordered the Oakwood from Lumber house
------------
Audio File: 74
Label: nonsensitive
Recognized Text: Lynn remember the order we received from colors House 2 weeks ago
------------
Audio File: 75
Label: nonsensitive
Recognized Text: did they change the shipping terms do we still have to deliver the order or will they come here to pick it up
------------
Audio File: 76
Label: nonsensitive
Recognized Text: I do not think so since we do not have to make delivery until the 20th of this month another 15 days
------------
Audio File: 77
Label: nonsensitive
Recognized Text: I am here
------------
Audio File: 78
Label: nonsensitive
Recognized Text: we still have to take care of the shipping process and it is still going to Chicago
------------
Audio File: 79
Label: nonsensitive
Recognized Text: they need to be aware of the change
------------
Audio File: 80
Label: nonsensitive
Recognized Text: overtime already started last week with all the employees in the assembly Department working an average of 2 hours overtime per day
------------
Audio File: 81
Label: nonsensitive
Recognized Text: yes I would not like to hear people complain
------------
Audio File: 82
Label: nonsensitive
Recognized Text: welcome Bob
------------
Audio File: 83
Label: nonsensitive
Recognized Text: I would hate to deal with unhappy customers
------------
Audio File: 84
Label: nonsensitive
Recognized Text: even though it is sometimes very difficult to please everybody
------------
Audio File: 85
Label: nonsensitive
Recognized Text: we're here today to discuss ways of improving sales in rural Market areas
------------
Audio File: 86
Label: nonsensitive
Recognized Text: let's get started
------------
Audio File: 87
Label: nonsensitive
Recognized Text: I'd also like to introduce Margaret Simmons who recently joined our team
------------
Audio File: 88
Label: nonsensitive
Recognized Text: you are welcome Jane
------------
Audio File: 89
Label: nonsensitive
Recognized Text: thank you Mark
------------
Audio File: 90
Label: nonsensitive
Recognized Text: I'm afraid our national sales director and trusting can't be with us today
------------
Audio File: 91
Label: nonsensitive
Recognized Text: I will call colors house and tell them everything is set to go
------------
Audio File: 92
Label: nonsensitive
Recognized Text: if we are all here let's get started
------------
Audio File: 93
Label: nonsensitive
Recognized Text: I doubt that I will be able to relax even when I get home at the end of the day
------------
Audio File: 94
Label: nonsensitive
Recognized Text: me neither okay everything is set
------------
Audio File: 95
Label: nonsensitive
Recognized Text: it was a great seminar John gave us tons of information on how to deal with employees
------------
Audio File: 96
Label: nonsensitive
Recognized Text: first of all I'd like you to please join me in welcoming Jack Peterson how Southwest Area Sales vice president
------------
Audio File: 97
Label: nonsensitive
Recognized Text: write Tom over to you
------------
Audio File: 98
Label: nonsensitive
Recognized Text: thank you for having me I'm looking forward to today's meeting
------------
Audio File: 99
Label: nonsensitive
Recognized Text: it is really stressful to deal with unhappy customers
------------
Audio File: 100
Label: sensitive
Recognized Text: I am Anna
------------
Audio File: 101
Label: sensitive
Recognized Text: I will I am very interested in the subject of leadership
------------
Audio File: 102
Label: sensitive
Recognized Text: we began the meeting by approving the changes in our sales reporting system discussed on May 30th
------------
Audio File: 103
Label: sensitive
Recognized Text: you'll find a copy of the main ideas developed and discussed in these sessions in the photocopies in front of you
------------
Audio File: 104
Label: sensitive
Recognized Text: hi I am Tom
------------
Audio File: 105
Label: sensitive
Recognized Text: our sales teams need more accurate information on our customers
------------
Audio File: 106
Label: sensitive
Recognized Text: Unknown
------------
Audio File: 107
Label: sensitive
Recognized Text: thank you very much Jack
------------
Audio File: 108
Label: sensitive
Recognized Text: unfortunately we're running short of time we'll have to leave that to another time
------------
Audio File: 109
Label: sensitive
Recognized Text: yes I like it Prentice Hall always delivers good seminars
------------
Audio File: 110
Label: sensitive
Recognized Text: the meeting is closed
------------
Audio File: 111
Label: sensitive
Recognized Text: I'm afraid I can't agree with you
------------
Audio File: 112
Label: sensitive
Recognized Text: I suggest we give our rural sales teams more help with Advanced customer information reporting
------------
Audio File: 113
Label: sensitive
Recognized Text: the topic of the next session how to make positive impression on others and gain visibility and influence in the workplace seems to be really interesting
------------
Audio File: 114
Label: sensitive
Recognized Text: Unknown
------------
Audio File: 115
Label: sensitive
Recognized Text: you need to sign up for the learn how to delegate seminar
------------
Audio File: 116
Label: sensitive
Recognized Text: well we provide our city sales staff with database information on all of our larger clients
------------
Audio File: 117
Label: sensitive
Recognized Text: I need to find a way to create Harmony and cooperation within my department
------------
Audio File: 118
Label: sensitive
Recognized Text: I was promoted to the position of supervisor a few months ago
------------
Audio File: 119
Label: sensitive
Recognized Text: it can be quite a difficult situation sometimes it is the reason why I am here today
------------
Audio File: 120
Label: sensitive
Recognized Text: as you can see we are developing new methods to reach out to our rural customers
------------
Audio File: 121
Label: sensitive
Recognized Text: the results of this survey will be delivered to our sales teams
------------
Audio File: 122
Label: sensitive
Recognized Text: would you like to add anything Jennifer
------------
Audio File: 123
Label: sensitive
Recognized Text: yes nothing is more frustrating than sitting idle with nothing to do
------------
Audio File: 124
Label: sensitive
Recognized Text: we should be providing the same sort of knowledge on our rural customers to our sales staff there
------------
Audio File: 125
Label: sensitive
Recognized Text: write it looks as though we've covered the main items is there any other business
------------
Audio File: 126
Label: sensitive
Recognized Text: could you repeat that please
------------
Audio File: 127
Label: sensitive
Recognized Text: oh we produce office equipment such as calculators and fax machines how about yours
------------
Audio File: 128
Label: sensitive
Recognized Text: no they gave us extra time to fill the New Order
------------
Audio File: 129
Label: sensitive
Recognized Text: don't worry colors house is willing to pay an extra 25% for the change
------------
Audio File: 130
Label: sensitive
Recognized Text: oh here is the change of order from colors house
------------
Audio File: 131
Label: sensitive
Recognized Text: we can take pride in a job well done besides it makes our job easier also
------------
Audio File: 132
Label: sensitive
Recognized Text: is it still due on the 20th
------------
Audio File: 133
Label: sensitive
Recognized Text: it is okay then when do we have to ship the order
------------
Audio File: 134
Label: sensitive
Recognized Text: I bet all the customers want their orders now or as soon as possible
------------
Audio File: 135
Label: sensitive
Recognized Text: it is fine with me because I love this company and I want to see it prosper
------------
Audio File: 136
Label: sensitive
Recognized Text: I hope the customers appreciate our quick response time and the fact that we always jump through hoops to give them whatever they want
------------
Audio File: 137
Label: sensitive
Recognized Text: then I do not have to make any shipping changes other than changing the pickup date
------------
Audio File: 138
Label: sensitive
Recognized Text: first let's go over the report from the last meeting which was held on June 24th
------------
Audio File: 139
Label: sensitive
Recognized Text: he is the author of The Seven Habits of a good leader
------------
Audio File: 140
Label: sensitive
Recognized Text: Unknown
------------
Audio File: 141
Label: sensitive
Recognized Text: she is in Kobe at the moment developing how far is Salesforce
------------
Audio File: 142
Label: sensitive
Recognized Text: you can sign and fax it back to them after reviewing it
------------
Audio File: 143
Label: sensitive
Recognized Text: thanks for your help Lynn
------------
Audio File: 144
Label: sensitive
Recognized Text: may I also introduce my assistant Bob camp
------------

[ ]
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Define a function to extract features from audio data
def extract_features(audio_data):
    # Placeholder for feature extraction process
    # Replace this with your feature extraction code (e.g., MFCC, log-mel spectrogram, etc.)
    # For demonstration purposes, we'll use a random feature vector with 20 dimensions
    return np.random.rand(20)

# Read the dataset and preprocess audio data
processed_data, labels = read_dataset(dataset_folder)

# Extract features from the processed data
X = [extract_features(data) for data, _ in processed_data]
y = labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)

              precision    recall  f1-score   support

nonsensitive       0.72      0.86      0.78        21
   sensitive       0.25      0.12      0.17         8

    accuracy                           0.66        29
   macro avg       0.48      0.49      0.47        29
weighted avg       0.59      0.66      0.61        29


[ ]
!pip install googletrans==4.0.0-rc1

Requirement already satisfied: googletrans==4.0.0-rc1 in /usr/local/lib/python3.10/dist-packages (4.0.0rc1)
Requirement already satisfied: httpx==0.13.3 in /usr/local/lib/python3.10/dist-packages (from googletrans==4.0.0-rc1) (0.13.3)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (2024.2.2)
Requirement already satisfied: hstspreload in /usr/local/lib/python3.10/dist-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (2024.5.1)
Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (1.3.1)
Requirement already satisfied: chardet==3.* in /usr/local/lib/python3.10/dist-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (3.0.4)
Requirement already satisfied: idna==2.* in /usr/local/lib/python3.10/dist-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (2.10)
Requirement already satisfied: rfc3986<2,>=1.3 in /usr/local/lib/python3.10/dist-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (1.5.0)
Requirement already satisfied: httpcore==0.9.* in /usr/local/lib/python3.10/dist-packages (from httpx==0.13.3->googletrans==4.0.0-rc1) (0.9.1)
Requirement already satisfied: h11<0.10,>=0.8 in /usr/local/lib/python3.10/dist-packages (from httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) (0.9.0)
Requirement already satisfied: h2==3.* in /usr/local/lib/python3.10/dist-packages (from httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) (3.2.0)
Requirement already satisfied: hyperframe<6,>=5.2.0 in /usr/local/lib/python3.10/dist-packages (from h2==3.*->httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) (5.2.0)
Requirement already satisfied: hpack<4,>=3.0 in /usr/local/lib/python3.10/dist-packages (from h2==3.*->httpcore==0.9.*->httpx==0.13.3->googletrans==4.0.0-rc1) (3.0.0)

[ ]
from googletrans import Translator

# Initialize the translator
translator = Translator()

# Example English text to translate
english_text = "welcome"

# Translate English text to Hindi
translation = translator.translate(english_text, dest='hi')

# Print the translated text
print("Translated Text (Hindi):", translation.text)

Translated Text (Hindi): स्वागत

[ ]
from googletrans import Translator

# Function to translate text to Telugu
def translate_to_telugu(text):
    translator = Translator()
    translation = translator.translate(text, dest='te')
    return translation.text

# Main function
def main():
    # English text to translate
    english_text = "arthi what are you doing?"

    # Translate to Telugu
    telugu_translation = translate_to_telugu(english_text)
    print("Telugu translation:", telugu_translation)

if __name__ == "__main__":
    main()

Telugu translation: ఆర్థీ మీరు ఏమి చేస్తున్నారు?

[ ]
from googletrans import Translator

# Function to translate text to Punjabi
def translate_to_punjabi(text):
    translator = Translator()
    translation = translator.translate(text, dest='pa')
    return translation.text

# Main function
def main():
    # English text to translate
    english_text = "Hello, how are you?"

    # Translate to Punjabi
    punjabi_translation = translate_to_punjabi(english_text)
    print("Punjabi translation:", punjabi_translation)

if __name__ == "__main__":
    main()

Punjabi translation: ਹੈਲੋ ਤੁਸੀ ਕਿਵੇਂ ਹੋ?

[ ]
translated_dataset_folder = r'C:\Users\srikr\OneDrive\Desktop\to store translated data'


[ ]
pip install transformers

Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.40.2)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.14.0)
Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.20.3)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (1.25.2)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (24.0)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.1)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2023.12.25)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.31.0)
Requirement already satisfied: tokenizers<0.20,>=0.19 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.19.1)
Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.4.3)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers) (4.66.4)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (2023.6.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.19.3->transformers) (4.11.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.2.2)

[ ]
from transformers import MarianMTModel

# Load the MarianMT model
model_name = "Helsinki-NLP/opus-mt-en-hi"  # Example model for English to Hindi translation
model = MarianMTModel.from_pretrained(model_name)



[ ]

Start coding or generate with AI.

[ ]
import os
import librosa
import speech_recognition as sr
from googletrans import Translator

# Path to your English voice dataset folder
english_dataset_folder = '/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData'

# Path to store the translated dataset
translated_dataset_folder = r'C:\Users\srikr\OneDrive\Desktop\to store translated data'

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the translator
translator = Translator()

# Function to read audio files from the English dataset folder, perform speech recognition,
# and translate the recognized text to Hindi
def process_dataset(english_folder, translated_folder):
    if not os.path.exists(translated_folder):
        os.makedirs(translated_folder)

    for label in os.listdir(english_folder):
        label_english_folder = os.path.join(english_folder, label)
        label_translated_folder = os.path.join(translated_folder, label)
        if os.path.isdir(label_english_folder):
            if not os.path.exists(label_translated_folder):
                os.makedirs(label_translated_folder)

            for audio_file in os.listdir(label_english_folder):
                if audio_file.endswith('.wav'):
                    english_audio_path = os.path.join(label_english_folder, audio_file)
                    translated_audio_path = os.path.join(label_translated_folder, audio_file)

                    # Perform speech recognition
                    with sr.AudioFile(english_audio_path) as source:
                        audio_data = recognizer.record(source)
                        try:
                            english_text = recognizer.recognize_google(audio_data)
                        except sr.UnknownValueError:
                            english_text = "Unknown"
                        except sr.RequestError as e:
                            english_text = "Error: {0}".format(e)

                    # Translate English text to Hindi
                    translation = translator.translate(english_text, dest='hi')

                    # Save translated text to a file
                    with open(translated_audio_path[:-4] + '.txt', 'w', encoding='utf-8') as text_file:
                        text_file.write(translation.text)

                    # Copy the audio file to the translated folder
                    os.system("cp {} {}".format(english_audio_path, translated_audio_path))

# Process the English dataset and translate it to Hindi
process_dataset(english_dataset_folder, translated_dataset_folder)


[ ]
import os
import speech_recognition as sr
from googletrans import Translator

# Initialize the recognizer
recognizer = sr.Recognizer()

# Initialize the translator
translator = Translator()

# Function to transcribe speech from an audio file and translate it to Hindi
def translate_audio_to_hindi(audio_file_path):
    try:
        # Use the recognizer to transcribe speech from the audio file
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            # Recognize speech from the audio
            english_text = recognizer.recognize_google(audio_data)
            print("Recognized English Text:", english_text)

            # Translate English text to Hindi
            translation = translator.translate(english_text, src='en', dest='hi')
            hindi_text = translation.text
            print("Translated Hindi Text:", hindi_text)

            return hindi_text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))
    except Exception as e:
        print("Error:", e)
        return None

# Example usage
audio_file_path = "/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData/nonsensitive/102.wav"
translated_text = translate_audio_to_hindi(audio_file_path)

Recognized English Text: after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements
Translated Hindi Text: संक्षेप में उन परिवर्तनों को संशोधित करने के बाद जो हम ग्राहक सहायता में सुधार के बाद एक मंथन सत्र में चले गए

[ ]
# English to Telugu vocabulary mapping
english_to_telugu_mapping = {
    "so": "కాబట్టి",
    "if": "అయితే",
    "there": "అక్కడ",
    "is": "ఉంది",
    "nothing": "ఏమీ",
    "else": "ఇతర",
    "we": "మేము",
    "need": "అవసరం",
    "to": "కోసం",
    "discuss": "చర్చ",
    "let's": "చెప్పండి",
    "move": "మారండి",
    "on": "ముందుకు",
    "today's": "ఈ రోజున",
    "agenda": "అజెండా",
    "the": "అలా",
    "meeting": "మీటింగ్",
    "was": "ఉంది",
    "declared": "ఘోషించబడింది",
    "closed": "మూసివేయబడింది",
    "at": "లో",
    "11.30": "11.30",
    "hi": "హాయ్",
    "it's": "ఇది",
    "me": "నాకు",
    "let": "అనుకోండి",
    "just": "కేవలం",
    "summarize": "సంగ్రహించుట",
    "main": "ముఖ్య",
    "points": "పాయింట్లు",
    "of": "వెలువడించడానికి",
    "last": "చివరి",
    "thank": "ధన్యవాదాలు",
    "you": "మీరు",
    "Tom": "టామ్",
    "have": "కావాలి",
    "all": "అన్ని",
    "received": "స్వీకరించబడింది",
    "a": "ఒక",
    "copy": "కాపీ",
    "today's": "ఈ రోజున",
    "you": "మీరు",
    "should": "ఉండాలి",
    "sign": "సెన్నిపించడానికి",
    "up": "అప్",
    "for": "కోసం",
    "that": "అది",
    "seminar": "సెమినార్",
    "next": "తర్వాత",
    "year": "సంవత్సరం",
    "after": "తర్వాత",
    "briefly": "సులభంగా",
    "revising": "మరచిపోయిన",
    "changes": "మార్పులు",
    "that": "అది",
    "will": "చేస్తాయి",
    "take": "తీసుకోవడానికి",
    "place": "ప్రతిష్టన",
    "moved": "కదిలేయబడిన",
    "a": "ఒక",
    "brainstorming": "బ్రెయిన్‌స్టార్మింగ్",
    "session": "సెషన్",
    "concerning": "గురించి",
    "customer": "కస్టమర్",
    "support": "మద్దతు",
    "improvements": "మెరుగైంపులు",
    "before": "ముందు",
    "begin": "ప్రారంభించు",
    "report": "నివేదిక",
    "like": "ఇష్టము",
    "get": "పొందడానికి",
    "some": "కొన్ని",
    "ideas": "ఆలోచనలు",
    "from": "నుండి",
    "I": "నేను",
    "also": "కూడా",
    "need": "అవసరం",
    "learn": "నేర్చుకోవటానికి",
    "how": "ఎలా",
    "better": "మెరుగుపరచుకోవడానికి",
    "mg": "ఎమ్‌జి",
    "my": "నా",
    "workload": "కార్యభారం",
    "always": "సదా",
    "run": "రన్",
    "out": "తేలిక",
    "of": "లో",
    "time": "సమయం",
    "an": "ఒక",
    "advertising": "ప్రకటన",
    "campaign": "ప్రచారయోజన",
    "to": "కోసం",
    "focus": "ప్రాధాన్యం",
    "on": "పైకి",
    "their": "వాళ్ల",
    "particular": "ప్రత్యేక",
    "needs": "అవసరాలు",
    "Unknown": "తెలియని",
    "think": "అనుకుంటున్నాను",
    "rural": "గ్రామీణ",
    "customers": "గడ్డారు",
    "want": "కావాలి",
    "feel": "అనుభవించటానికి",
    "as": "వంటి",
    "important": "ముఖ్యమైన",
    "our": "మా",
    "living": "జీవించడం",
    "in": "లో",
    "cities": "నగరాలు",
    "well": "బాగా",
    "me": "నాకు",
    "begin": "ప్రారంభించు",
    "with": "తో",
    "this": "ఇది",
    "powerpoint": "పవర్‌పాయింట్",
    "presentation": "ప్రాధర్యము",
    "Jack": "జాక్",
    "presents": "ప్రదర్శిస్తాడు",
    "his": "ఆతని",
    "how": "ఎలా",
    "feel": "అనుభవించండి",
    "about": "గురించి",
    "sales": "అమ్మకాలు",
    "your": "మీ",
    "districts": "జిల్లాలు",
    "excuse": "క్షమాపణ",
    "me": "నాకు",
    "I": "నేను",
    "didn't": "లేదు",
    "catch": "అర్థం",
    "that": "అది",
    "can": "చెయ్యవచ్చు",
    "we": "మేము",
    "fix": "మార్పు",
    "the": "ది",
    "next": "తర్వాత",
    "meeting": "మీటింగ్",
    "please": "దయచేసి",
    "suggest": "సూచించుకుంద",
}


# Function to map English words to Telugu
def map_to_telugu(text):
    words = text.split()
    telugu_text = ' '.join([english_to_telugu_mapping.get(word.lower(), word) for word in words])
    return telugu_text

# Test vocabulary mapping for a sample text
sample_text = " after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements"
telugu_translation = map_to_telugu(sample_text)
print("Telugu translation:", telugu_translation)

Telugu translation: తర్వాత సులభంగా మరచిపోయిన ది మార్పులు అది చేస్తాయి తీసుకోవడానికి ప్రతిష్టన మేము కదిలేయబడిన పైకి కోసం ఒక బ్రెయిన్‌స్టార్మింగ్ సెషన్ గురించి తర్వాత కస్టమర్ మద్దతు మెరుగైంపులు

[ ]
english_to_punjabi_mapping = {
    "so": "ਤਾਂ",
    "if": "ਜੇ",
    "there": "ਉੱਥੇ",
    "is": "ਹੈ",
    "nothing": "ਕੁਝ ਨਹੀਂ",
    "else": "ਹੋਰ",
    "we": "ਅਸੀਂ",
    "need": "ਲੋੜ",
    "to": "ਨੂੰ",
    "discuss": "ਚਰਚਾ ਕਰਨਾ",
    "let's": "ਚੱਲੋ",
    "move": "ਚੱਲੋ",
    "on": "ਉੱਤੇ",
    "today's": "ਅੱਜ ਦੇ",
    "agenda": "ਐਜੰਡਾ",
    "the": "ਉਹ",
    "meeting": "ਮੀਟਿੰਗ",
    "was": "ਸੀ",
    "declared": "ਘੋਸ਼ਿਤ",
    "closed": "ਬੰਦ",
    "at": "ਤੇ",
    "11.30": "11.30",
    "hi": "ਹਾਂ",
    "it's": "ਇਹ",
    "me": "ਮੈਨੂੰ",
    "let": "ਦੱਸੋ",
    "just": "ਸਿਰਫ",
    "summarize": "ਸੰਖੇਪਿਕ",
    "main": "ਮੁੱਖ",
    "points": "ਬਿੰਦੂ",
    "of": "ਦਾ",
    "last": "ਆਖਰੀ",
    "thank": "ਧੰਨਵਾਦ",
    "you": "ਤੁਸੀਂ",
    "Tom": "ਟਾਮ",
    "have": "ਹੈ",
    "all": "ਸਭ",
    "received": "ਪ੍ਰਾਪਤ",
    "a": "ਇੱਕ",
    "copy": "ਨਕਲ",
    "should": "ਚਾਹੀਦਾ ਹੈ",
    "sign": "ਹਸਤਾਖਰਾਰ",
    "up": "ਉੱਤੇ",
    "for": "ਲਈ",
    "that": "ਉਹ",
    "seminar": "ਸੇਮੀਨਾਰ",
    "next": "ਅਗਲੇ",
    "year": "ਸਾਲ",
    "after": "ਬਾਅਦ",
    "briefly": "ਛੋਟੇ ਅਵਧੀ",
    "revising": "ਮਰੀਜ਼",
    "changes": "ਤਬਦੀਲੀਆਂ",
    "will": "ਹੋਵੇਗਾ",
    "take": "ਲਓ",
    "place": "ਸਥਾਨ",
    "moved": "ਹਿਲਾਇਆ",
    "After": "ਬਾਅਦ",
    "briefly": "ਥੋੜ੍ਹਾ",
    "revising": "ਸਮੀਖਿਆ",
    "the": "ਵਾਲੀ",
    "changes": "ਬਦਲਾਓ",
    "that": "ਜੋ",
    "will": "ਕਰੇਗੀ",
    "take": "ਲਓ",
    "place": "ਜਗ੍ਹਾ",
    "we": "ਅਸੀਂ",
    "moved": "ਚੱਲੇ",
    "on": "ਉੱਤੇ",
    "to": "ਨੂੰ",
    "a": "ਇੱਕ",
    "brainstorming": "ਬ੍ਰੇਨਸਟਰਮਿੰਗ",
    "session": "ਸੈਸ਼ਨ",
    "concerning": "ਬਾਰੇ",
    "customer": "ਗਾਹਕ",
    "support": "ਸਹਿਯੋਗ",
    "improvements": "ਸੁਧਾਰ",
    }

# Sample English sentence
english_sentence = "after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements"

# Split the sentence into words
words = english_sentence.split()

# Translate each word using the mapping
punjabi_translation = ' '.join(english_to_punjabi_mapping.get(word, word) for word in words)

# Print the Punjabi translation
print("Punjabi Translation:", punjabi_translation)

Punjabi Translation: ਬਾਅਦ ਥੋੜ੍ਹਾ ਸਮੀਖਿਆ ਵਾਲੀ ਬਦਲਾਓ ਜੋ ਕਰੇਗੀ ਲਓ ਜਗ੍ਹਾ ਅਸੀਂ ਚੱਲੇ ਉੱਤੇ ਨੂੰ ਇੱਕ ਬ੍ਰੇਨਸਟਰਮਿੰਗ ਸੈਸ਼ਨ ਬਾਰੇ ਬਾਅਦ ਗਾਹਕ ਸਹਿਯੋਗ ਸੁਧਾਰ

[ ]
# English to Kannada vocabulary mapping
vocabulary_mapping = {
    "so": "ಹೌದು",
    "if": "ಹೇಗಾದರೆ",
    "there": "ಅಲ್ಲಿ",
    "is": "ಇದೆ",
    "nothing": "ಯಾವುದೂ ಇಲ್ಲ",
    "else": "ಇನ್ನೊಂದು",
    "we": "ನಾವು",
    "need": "ಬೇಕು",
    "to": "ಗೆ",
    "discuss": "ಚರ್ಚಿಸಬೇಕಾಗಿದೆ",
    "let's": "ನೋಡು",
    "move": "ಹೆಜ್ಜೆ",
    "on": "ಮೇಲೆ",
    "today's": "ಇಂದಿನ",
    "agenda": "ಕಾರ್ಯಾಚರಣೆ",
    "the": "ಅದು",
    "meeting": "ಭೇಟಿ",
    "was": "ಆಗಿತ್ತು",
    "declared": "ಘೋಷಿಸಲಾಗಿತ್ತು",
    "closed": "ಮುಚ್ಚಲಾಗಿತ್ತು",
    "at": "ನಲ್ಲಿ",
    "hi": "ನಮಸ್ಕಾರ",
    "it's": "ಇದು",
    "me": "ನಾನು",
    "let": "ಬಿಟ್ಟು",
    "just": "ಸರಿ",
    "summarize": "ಸಾರಿಸು",
    "main": "ಮುಖ್ಯ",
    "points": "ಅಂಶಗಳು",
    "of": "ನ",
    "last": "ಕೊನೆಯ",
    "report": "ವರದಿ",
    "thank": "ಧನ್ಯವಾದ",
    "you": "ನೀವು",
    "tom": "ಟಾಮ್",
    "have": "ಹೊಂದಿದ್ದೇನೆ",
    "all": "ಎಲ್ಲರೂ",
    "received": "ಸ್ವೀಕರಿಸಿದ",
    "a": "ಒಂದು",
    "copy": "ನಕಲಿ",
    "of": "ನ",
    "today's": "ಇಂದಿನ",
    "you": "ನೀವು",
    "should": "ಬೇಕು",
    "sign": "ಸಹಿ",
    "up": "ಮೇಲೇ",
    "for": "ಗೆ",
    "that": "ಅದು",
    "seminar": "ಸಮಿನಾರ್",
    "next": "ಮುಂದಿನ",
    "year": "ವರ್ಷ",
    "after": "ನಂತರ",
    "briefly": "ಸಂಕ್ಷೇಪವಾಗಿ",
    "revising": "ಪುನರ್ವಿಮರ್ಶೆ",
    "the": "ಅದು",
    "changes": "ಬದಲಾವಣೆಗಳು",
    "that": "ಅದು",
    "will": "ಸಾಧ್ಯ",
    "take": "ತೆಗೆದುಕೊಳ್ಳುತ್ತವೆ",
    "place": "ಸ್ಥಳ",
    "we": "ನಾವು",
    "moved": "ಹಾರಾಡಿದ್ದೆವು",
    "on": "ಮೇಲೆ",
    "to": "ಗೆ",
    "a": "ಒಂದು",
    "brainstorming": "ಭಾವನಾ ವೃತ್ತಿ",
    "session": "ಸೆಷನ್",
    "concerning": "ಪ್ರಸಂಗದ",
    "after": "ನಂತರ",
    "customer": "ಗ್ರಾಹಕ",
    "support": "ಬೆಂಬಲ",
    "improvements": "ಮೆಚ್ಚಿಸುವಂತೆ",
    "before": "ಹಿಂದೆ",
    "i": "ನಾನು",
    "begin": "ಪ್ರಾರಂಭಿಸುತ್ತಿದ್ದೇನೆ",
    "the": "ಅದು",
    "report": "ವರದಿ",
    "i": "ನಾನು",
    "like": "ಇಷ್ಟು",
    "to": "ಗೆ",
    "get": "ಪಡೆಯಲು",
    "some": "ಕೆಲವು",
    "ideas": "ಕಲ್ಪನೆಗಳು",
    "from": "ಇಂದ",
    "you": "ನೀವು",
    "all": "ಎಲ್ಲರೂ",
    "unknown": "ಅಜ್ಞಾತ",
    "i": "ನಾನು",
    "think": "ಭಾವಿಸುತ್ತೇನೆ",
    "rural": "ಗ್ರಾಮೀಣ",
    "customers": "ಗ್ರಾಹಕರು",
    "want": "ಬಯಸುತ್ತಾರೆ",
    "to": "ಗೆ",
    "feel": "ಅನುಭವಿಸು",
    "as": "ಅಂತಹ",
    "important": "ಮುಖ್ಯ",
    "our": "ನಮ್ಮ",
    "living": "ಬದುಕು",
    "in": "ಇಲ್ಲ",
    "cities": "ನಗರಗಳು",
    "well": "ಚೆನ್ನಾಗಿ",
    "me": "ನನಗೆ",
    "with": "ಜೊತೆ",
    "this": "ಈ",
    "powerpoint": "ಪವರ್‌ಪಾಯಿಂಟ್",
    "presentation": "ಪ್ರದರ್ಶನ",
    "jack": "ಜ್ಯಾಕ್",
    "presents": "ಪ್ರದರ್ಶಿಸುತ್ತಾನೆ",
    "his": "ಅವನ",
    "how": "ಹೇಗೆ",
    "do": "ಮಾಡು",
    "feel": "ಭಾವಿಸು",
    "about": "ಬಗ್ಗೆ",
    "rural": "ಗ್ರಾಮೀಣ",
    "sales": "ಮಾರಾಟದ",
    "in": "ಇಲ್ಲ",
    "your": "ನಿಮ್ಮ",
    "districts": "ಜಿಲ್ಲೆಗಳು",
    "excuse": "ಕ್ಷಮಿಸಿ",
    "me": "ನನಗೆ",
    "i": "ನಾನು",
    "didn't": "ಇಲ್ಲಿರಲಿಲ್ಲ",
    "catch": "ಹಿಡಿಯಲು",
    "that": "ಅದು",
    "can": "ಸಾಧ್ಯ",
    "we": "ನಾವು",
    "fix": "ಮಾಡು",
    "the": "ಅದು",
    "next": "ಮುಂದಿನ",
    "meeting": "ಭೇಟಿ",
    "please": "ದಯವಿಟ್ಟು",
    "i": "ನಾನು",
    "suggest": "ಸೂಚಿಸು",
    "we": "ನಾವು",
    "break": "ವಿರಾಮ",
    "up": "ಮೇಲೇ",
    "into": "ಗೆ",
    "groups": "ಗುಂಪುಗಳು",
    "and": "ಮತ್ತು",
    "ideas": "ಕಲ್ಪನೆಗಳು",
    "we've": "ನಮಗೆ",
    "seen": "ನೋಡಿದ್ದೇವೆ",
    "presented": "ಪ್ರಸ್ತುತಿಸಲಾಗಿದೆ",
    "we": "ನಾವು",
    "are": "ಇರುವುದು",
    "considering": "ಪರಿಗಣಿಸುತ್ತಿದ್ದೇವೆ",
    "specific": "ನಿರ್ದಿಷ್ಟ",
    "data": "ಡೇಟಾ",
    "mining": "ಖನಿ",
    "procedures": "ನಿಯಮಗಳು",
    "to": "ಗೆ",
    "help": "ಸಹಾಯ",
    "deepen": "ಆಳವಾಗಿಸಲು",
    "our": "ನಮ್ಮ",
    "understanding": "ಅರಿವು",
    "i'd": "ನಾನು",
    "like": "ಇಷ್ಟು",
    "to": "ಗೆ",
    "thank": "ಧನ್ಯವಾದ",
    "jack": "ಜ್ಯಾಕ್",
    "for": "ಗೆ",
    "coming": "ಬರುವುದು",
    "our": "ನಮ್ಮ",
    "meeting": "ಭೇಟಿ",
    "today": "ಇಂದ",
    "and": "ಮತ್ತು",
    "i": "ನಾನು",
    "have": "ಹೊಂದಿದ್ದೇನೆ",
    "to": "ಗೆ",
    "a": "ಒಂದು",
    "thought": "ಭಾವನೆ",
    "let's": "ನೋಡು",
    "start": "ಪ್ರಾರಂಭಿಸು",
    "going": "ಹೋಗುವುದು",
    "around": "ಸುತ್ತ",
    "the": "ಅದು",
    "table": "ಟೇಬಲ್",
    "gathering": "ಸಮೂಹ",
    "your": "ನಿಮ್ಮ",
    "feedback": "ಪ್ರತಿಸ್ಪಂದನೆ",
    "on": "ಮೇಲೆ",
    "last": "ಕೊನೆಯ",
    "week's": "ವಾರದ",
    "presentation": "ಪ್ರದರ್ಶನ",
    "do": "ಮಾಡು",
    "you": "ನೀವು",
    "all": "ಎಲ್ಲರೂ",
    "have": "ಹೊಂದಿದ್ದೇನೆ",
    "any": "ಯಾವುದೇ",
    "other": "ಬೇರೆ",
    "questions": "ಪ್ರಶ್ನೆಗಳು",
    "before": "ಹಿಂದೆ",
    "we": "ನಾವು",
    "move": "ಹೆಜ್ಜೆ",
    "on": "ಮೇಲೆ",
    "to": "ಗೆ",
    "our": "ನಮ್ಮ",
    "next": "ಮುಂದಿನ",
    "topic": "ವಿಷಯ",
    "this": "ಈ",
    "is": "ಇದೆ",
    "important": "ಮುಖ್ಯ",
    "i": "ನಾನು",
    "think": "ಭಾವಿಸುತ್ತೇನೆ",
    "your": "ನಿಮ್ಮ",
    "feedback": "ಪ್ರತಿಸ್ಪಂದನೆ",
    "about": "ಬಗ್ಗೆ",
    "this": "ಈ",
    "matter": "ವಿಷಯ",
    "would": "ಇರಬಹುದು",
    "be": "ಆಗಬಹುದು",
    "appreciated": "ಮೆಚ್ಚಿದ",
    "what": "ಏನು",
    "are": "ಇರುವುದು",
    "your": "ನಿಮ್ಮ",
    "initial": "ಮೊದಲಿನ",
    "thoughts": "ಭಾವನೆಗಳು",
    "on": "ಮೇಲೆ",
    "the": "ಅದು",
    "proposal": "ಸೂಚನೆ",
    "i": "ನಾನು",
    "look": "ನೋಡು",
    "forward": "ಮುನ್ನಡೆ",
    "to": "ಗೆ",
    "hearing": "ಕೇಳುತ್ತಿದ್ದೇನೆ",
    "from": "ಇಂದ",
    "you": "ನೀವು",
    "all": "ಎಲ್ಲರೂ"
}

# Function to translate English sentence to Kannada
def translate_to_kannada(sentence):
    words = sentence.split()
    translated_sentence = ' '.join([vocabulary_mapping.get(word.lower(), word) for word in words])
    return translated_sentence

# Example sentence to translate
english_sentence = "after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements"

# Translating the English sentence to Kannada
kannada_translation = translate_to_kannada(english_sentence)
print("English Sentence:", english_sentence)
print("Kannada Translation:", kannada_translation)

English Sentence: after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements
Kannada Translation: ನಂತರ ಸಂಕ್ಷೇಪವಾಗಿ ಪುನರ್ವಿಮರ್ಶೆ ಅದು ಬದಲಾವಣೆಗಳು ಅದು ಸಾಧ್ಯ ತೆಗೆದುಕೊಳ್ಳುತ್ತವೆ ಸ್ಥಳ ನಾವು ಹಾರಾಡಿದ್ದೆವು ಮೇಲೆ ಗೆ ಒಂದು ಭಾವನಾ ವೃತ್ತಿ ಸೆಷನ್ ಪ್ರಸಂಗದ ನಂತರ ಗ್ರಾಹಕ ಬೆಂಬಲ ಮೆಚ್ಚಿಸುವಂತೆ

[ ]
from sklearn.metrics import classification_report

# Actual and predicted labels
actual_labels = ['Telugu', 'Telugu', 'Punjabi', 'Kannada', 'Telugu', 'Punjabi', 'Punjabi', 'Kannada', 'Kannada', 'Kannada']
predicted_labels = ['Telugu', 'Kannada', 'Punjabi', 'Kannada', 'Telugu', 'Punjabi', 'Kannada', 'Kannada', 'Kannada', 'Punjabi']

# Generate classification report
report = classification_report(actual_labels, predicted_labels)

# Print the classification report
print(report)

              precision    recall  f1-score   support

     Kannada       0.60      0.75      0.67         4
     Punjabi       0.67      0.67      0.67         3
      Telugu       1.00      0.67      0.80         3

    accuracy                           0.70        10
   macro avg       0.76      0.69      0.71        10
weighted avg       0.74      0.70      0.71        10


[ ]

Start coding or generate with AI.
Colab paid products - Cancel contracts here
t os

# Mount Google Drive
drive.mount('/content/drive')


import os
import librosa

# Path to your dataset folder
dataset_folder = '/content/nonsensitive'

# Function to read audio files from the dataset folder
def read_dataset(folder):
    dataset = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for audio_file in os.listdir(label_folder):
                if audio_file.endswith('.wav'):  # Assuming your audio files are in WAV format
                    file_path = os.path.join(label_folder, audio_file)
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    dataset.append(audio_data)
                    labels.append(label)
    return dataset, labels

# Read the dataset
data, labels = read_dataset(dataset_folder)

# Example: print the number of audio files and their corresponding labels
print("Number of audio files:", len(data))
print("Labels:", labels)


from collections import Counter


# Count the occurrences of each label
label_counts = Counter(labels)

# Print the label counts
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")


import os
import librosa
import numpy as np

# Define the function for preprocessing audio files
def preprocess_audio_folder(folder_path, sample_rate=16000, duration=2):
    preprocessed_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            audio_file = os.path.join(folder_path, filename)
            preprocessed_audio = preprocess_audio(audio_file, sample_rate, duration)
            preprocessed_data.append((preprocessed_audio, filename))  # Store preprocessed audio and filename
    return preprocessed_data

# Function to preprocess a single audio file
def preprocess_audio(audio_file, sample_rate=16000, duration=2):
    # Load audio file
    audio_data, _ = librosa.load(audio_file, sr=sample_rate, duration=duration)

    # Trim silence
    audio_data, _ = librosa.effects.trim(audio_data)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)

    # Pad or truncate MFCCs to a fixed length
    max_len = 100
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]

    return mfccs

# Example usage
folder_path = '/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData'
preprocessed_data = preprocess_audio_folder(folder_path)

# Example print the preprocessed data
for audio_data, filename in preprocessed_data:
    print("Filename:", filename)
    print("Preprocessed Data Shape:", audio_data.shape)
    # Use preprocessed data for further processing or model training


pip install SpeechRecognition


import speech_recognition as sr


import os
import librosa
import speech_recognition as sr

# Path to your dataset folder
dataset_folder = '/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData'

# Function to read audio files from the dataset folder
def read_dataset(folder):
    dataset = []
    labels = []
    recognizer = sr.Recognizer()
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for audio_file in os.listdir(label_folder):
                if audio_file.endswith('.wav'):  # Assuming your audio files are in WAV format
                    file_path = os.path.join(label_folder, audio_file)
                    audio_data, sample_rate = librosa.load(file_path, sr=None)
                    text = recognize_audio(recognizer, file_path)
                    dataset.append((audio_data, text))
                    labels.append(label)
    return dataset, labels

# Function to recognize speech from an audio file
def recognize_audio(recognizer, file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Unknown"
        except sr.RequestError as e:
            return "Error: {0}".format(e)

# Read the dataset
data, labels = read_dataset(dataset_folder)

# Example: print the number of audio files, their corresponding labels, and recognized text
for i in range(len(data)):
    print("Audio File:", i+1)
    print("Label:", labels[i])
    print("Recognized Text:", data[i][1])
    print("------------")


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Define a function to extract features from audio data
def extract_features(audio_data):
    # Placeholder for feature extraction process
    # Replace this with your feature extraction code (e.g., MFCC, log-mel spectrogram, etc.)
    # For demonstration purposes, we'll use a random feature vector with 20 dimensions
    return np.random.rand(20)

# Read the dataset and preprocess audio data
processed_data, labels = read_dataset(dataset_folder)

# Extract features from the processed data
X = [extract_features(data) for data, _ in processed_data]
y = labels

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)

# Print the classification report
print(report)


!pip install googletrans==4.0.0-rc1


from googletrans import Translator

# Initialize the translator
translator = Translator()

# Example English text to translate
english_text = "welcome"

# Translate English text to Hindi
translation = translator.translate(english_text, dest='hi')

# Print the translated text
print("Translated Text (Hindi):", translation.text)


from googletrans import Translator

# Function to translate text to Telugu
def translate_to_telugu(text):
    translator = Translator()
    translation = translator.translate(text, dest='te')
    return translation.text

# Main function
def main():
    # English text to translate
    english_text = "arthi what are you doing?"

    # Translate to Telugu
    telugu_translation = translate_to_telugu(english_text)
    print("Telugu translation:", telugu_translation)

if __name__ == "__main__":
    main()


from googletrans import Translator

# Function to translate text to Punjabi
def translate_to_punjabi(text):
    translator = Translator()
    translation = translator.translate(text, dest='pa')
    return translation.text

# Main function
def main():
    # English text to translate
    english_text = "Hello, how are you?"

    # Translate to Punjabi
    punjabi_translation = translate_to_punjabi(english_text)
    print("Punjabi translation:", punjabi_translation)

if __name__ == "__main__":
    main()


translated_dataset_folder = r'C:\Users\srikr\OneDrive\Desktop\to store translated data'


pip install transformers


from transformers import MarianMTModel

# Load the MarianMT model
model_name = "Helsinki-NLP/opus-mt-en-hi"  # Example model for English to Hindi translation
model = MarianMTModel.from_pretrained(model_name)




import os
import librosa
import speech_recognition as sr
from googletrans import Translator

# Path to your English voice dataset folder
english_dataset_folder = '/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData'

# Path to store the translated dataset
translated_dataset_folder = r'C:\Users\srikr\OneDrive\Desktop\to store translated data'

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Initialize the translator
translator = Translator()

# Function to read audio files from the English dataset folder, perform speech recognition,
# and translate the recognized text to Hindi
def process_dataset(english_folder, translated_folder):
    if not os.path.exists(translated_folder):
        os.makedirs(translated_folder)

    for label in os.listdir(english_folder):
        label_english_folder = os.path.join(english_folder, label)
        label_translated_folder = os.path.join(translated_folder, label)
        if os.path.isdir(label_english_folder):
            if not os.path.exists(label_translated_folder):
                os.makedirs(label_translated_folder)

            for audio_file in os.listdir(label_english_folder):
                if audio_file.endswith('.wav'):
                    english_audio_path = os.path.join(label_english_folder, audio_file)
                    translated_audio_path = os.path.join(label_translated_folder, audio_file)

                    # Perform speech recognition
                    with sr.AudioFile(english_audio_path) as source:
                        audio_data = recognizer.record(source)
                        try:
                            english_text = recognizer.recognize_google(audio_data)
                        except sr.UnknownValueError:
                            english_text = "Unknown"
                        except sr.RequestError as e:
                            english_text = "Error: {0}".format(e)

                    # Translate English text to Hindi
                    translation = translator.translate(english_text, dest='hi')

                    # Save translated text to a file
                    with open(translated_audio_path[:-4] + '.txt', 'w', encoding='utf-8') as text_file:
                        text_file.write(translation.text)

                    # Copy the audio file to the translated folder
                    os.system("cp {} {}".format(english_audio_path, translated_audio_path))

# Process the English dataset and translate it to Hindi
process_dataset(english_dataset_folder, translated_dataset_folder)


import os
import speech_recognition as sr
from googletrans import Translator

# Initialize the recognizer
recognizer = sr.Recognizer()

# Initialize the translator
translator = Translator()

# Function to transcribe speech from an audio file and translate it to Hindi
def translate_audio_to_hindi(audio_file_path):
    try:
        # Use the recognizer to transcribe speech from the audio file
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            # Recognize speech from the audio
            english_text = recognizer.recognize_google(audio_data)
            print("Recognized English Text:", english_text)

            # Translate English text to Hindi
            translation = translator.translate(english_text, src='en', dest='hi')
            hindi_text = translation.text
            print("Translated Hindi Text:", hindi_text)

            return hindi_text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Speech Recognition service; {0}".format(e))
    except Exception as e:
        print("Error:", e)
        return None

# Example usage
audio_file_path = "/content/drive/MyDrive/voice dataset NLP/TextDataset/VoiceData/nonsensitive/102.wav"
translated_text = translate_audio_to_hindi(audio_file_path)


# English to Telugu vocabulary mapping
english_to_telugu_mapping = {
    "so": "కాబట్టి",
    "if": "అయితే",
    "there": "అక్కడ",
    "is": "ఉంది",
    "nothing": "ఏమీ",
    "else": "ఇతర",
    "we": "మేము",
    "need": "అవసరం",
    "to": "కోసం",
    "discuss": "చర్చ",
    "let's": "చెప్పండి",
    "move": "మారండి",
    "on": "ముందుకు",
    "today's": "ఈ రోజున",
    "agenda": "అజెండా",
    "the": "అలా",
    "meeting": "మీటింగ్",
    "was": "ఉంది",
    "declared": "ఘోషించబడింది",
    "closed": "మూసివేయబడింది",
    "at": "లో",
    "11.30": "11.30",
    "hi": "హాయ్",
    "it's": "ఇది",
    "me": "నాకు",
    "let": "అనుకోండి",
    "just": "కేవలం",
    "summarize": "సంగ్రహించుట",
    "main": "ముఖ్య",
    "points": "పాయింట్లు",
    "of": "వెలువడించడానికి",
    "last": "చివరి",
    "thank": "ధన్యవాదాలు",
    "you": "మీరు",
    "Tom": "టామ్",
    "have": "కావాలి",
    "all": "అన్ని",
    "received": "స్వీకరించబడింది",
    "a": "ఒక",
    "copy": "కాపీ",
    "today's": "ఈ రోజున",
    "you": "మీరు",
    "should": "ఉండాలి",
    "sign": "సెన్నిపించడానికి",
    "up": "అప్",
    "for": "కోసం",
    "that": "అది",
    "seminar": "సెమినార్",
    "next": "తర్వాత",
    "year": "సంవత్సరం",
    "after": "తర్వాత",
    "briefly": "సులభంగా",
    "revising": "మరచిపోయిన",
    "changes": "మార్పులు",
    "that": "అది",
    "will": "చేస్తాయి",
    "take": "తీసుకోవడానికి",
    "place": "ప్రతిష్టన",
    "moved": "కదిలేయబడిన",
    "a": "ఒక",
    "brainstorming": "బ్రెయిన్‌స్టార్మింగ్",
    "session": "సెషన్",
    "concerning": "గురించి",
    "customer": "కస్టమర్",
    "support": "మద్దతు",
    "improvements": "మెరుగైంపులు",
    "before": "ముందు",
    "begin": "ప్రారంభించు",
    "report": "నివేదిక",
    "like": "ఇష్టము",
    "get": "పొందడానికి",
    "some": "కొన్ని",
    "ideas": "ఆలోచనలు",
    "from": "నుండి",
    "I": "నేను",
    "also": "కూడా",
    "need": "అవసరం",
    "learn": "నేర్చుకోవటానికి",
    "how": "ఎలా",
    "better": "మెరుగుపరచుకోవడానికి",
    "mg": "ఎమ్‌జి",
    "my": "నా",
    "workload": "కార్యభారం",
    "always": "సదా",
    "run": "రన్",
    "out": "తేలిక",
    "of": "లో",
    "time": "సమయం",
    "an": "ఒక",
    "advertising": "ప్రకటన",
    "campaign": "ప్రచారయోజన",
    "to": "కోసం",
    "focus": "ప్రాధాన్యం",
    "on": "పైకి",
    "their": "వాళ్ల",
    "particular": "ప్రత్యేక",
    "needs": "అవసరాలు",
    "Unknown": "తెలియని",
    "think": "అనుకుంటున్నాను",
    "rural": "గ్రామీణ",
    "customers": "గడ్డారు",
    "want": "కావాలి",
    "feel": "అనుభవించటానికి",
    "as": "వంటి",
    "important": "ముఖ్యమైన",
    "our": "మా",
    "living": "జీవించడం",
    "in": "లో",
    "cities": "నగరాలు",
    "well": "బాగా",
    "me": "నాకు",
    "begin": "ప్రారంభించు",
    "with": "తో",
    "this": "ఇది",
    "powerpoint": "పవర్‌పాయింట్",
    "presentation": "ప్రాధర్యము",
    "Jack": "జాక్",
    "presents": "ప్రదర్శిస్తాడు",
    "his": "ఆతని",
    "how": "ఎలా",
    "feel": "అనుభవించండి",
    "about": "గురించి",
    "sales": "అమ్మకాలు",
    "your": "మీ",
    "districts": "జిల్లాలు",
    "excuse": "క్షమాపణ",
    "me": "నాకు",
    "I": "నేను",
    "didn't": "లేదు",
    "catch": "అర్థం",
    "that": "అది",
    "can": "చెయ్యవచ్చు",
    "we": "మేము",
    "fix": "మార్పు",
    "the": "ది",
    "next": "తర్వాత",
    "meeting": "మీటింగ్",
    "please": "దయచేసి",
    "suggest": "సూచించుకుంద",
}


# Function to map English words to Telugu
def map_to_telugu(text):
    words = text.split()
    telugu_text = ' '.join([english_to_telugu_mapping.get(word.lower(), word) for word in words])
    return telugu_text

# Test vocabulary mapping for a sample text
sample_text = " after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements"
telugu_translation = map_to_telugu(sample_text)
print("Telugu translation:", telugu_translation)


english_to_punjabi_mapping = {
    "so": "ਤਾਂ",
    "if": "ਜੇ",
    "there": "ਉੱਥੇ",
    "is": "ਹੈ",
    "nothing": "ਕੁਝ ਨਹੀਂ",
    "else": "ਹੋਰ",
    "we": "ਅਸੀਂ",
    "need": "ਲੋੜ",
    "to": "ਨੂੰ",
    "discuss": "ਚਰਚਾ ਕਰਨਾ",
    "let's": "ਚੱਲੋ",
    "move": "ਚੱਲੋ",
    "on": "ਉੱਤੇ",
    "today's": "ਅੱਜ ਦੇ",
    "agenda": "ਐਜੰਡਾ",
    "the": "ਉਹ",
    "meeting": "ਮੀਟਿੰਗ",
    "was": "ਸੀ",
    "declared": "ਘੋਸ਼ਿਤ",
    "closed": "ਬੰਦ",
    "at": "ਤੇ",
    "11.30": "11.30",
    "hi": "ਹਾਂ",
    "it's": "ਇਹ",
    "me": "ਮੈਨੂੰ",
    "let": "ਦੱਸੋ",
    "just": "ਸਿਰਫ",
    "summarize": "ਸੰਖੇਪਿਕ",
    "main": "ਮੁੱਖ",
    "points": "ਬਿੰਦੂ",
    "of": "ਦਾ",
    "last": "ਆਖਰੀ",
    "thank": "ਧੰਨਵਾਦ",
    "you": "ਤੁਸੀਂ",
    "Tom": "ਟਾਮ",
    "have": "ਹੈ",
    "all": "ਸਭ",
    "received": "ਪ੍ਰਾਪਤ",
    "a": "ਇੱਕ",
    "copy": "ਨਕਲ",
    "should": "ਚਾਹੀਦਾ ਹੈ",
    "sign": "ਹਸਤਾਖਰਾਰ",
    "up": "ਉੱਤੇ",
    "for": "ਲਈ",
    "that": "ਉਹ",
    "seminar": "ਸੇਮੀਨਾਰ",
    "next": "ਅਗਲੇ",
    "year": "ਸਾਲ",
    "after": "ਬਾਅਦ",
    "briefly": "ਛੋਟੇ ਅਵਧੀ",
    "revising": "ਮਰੀਜ਼",
    "changes": "ਤਬਦੀਲੀਆਂ",
    "will": "ਹੋਵੇਗਾ",
    "take": "ਲਓ",
    "place": "ਸਥਾਨ",
    "moved": "ਹਿਲਾਇਆ",
    "After": "ਬਾਅਦ",
    "briefly": "ਥੋੜ੍ਹਾ",
    "revising": "ਸਮੀਖਿਆ",
    "the": "ਵਾਲੀ",
    "changes": "ਬਦਲਾਓ",
    "that": "ਜੋ",
    "will": "ਕਰੇਗੀ",
    "take": "ਲਓ",
    "place": "ਜਗ੍ਹਾ",
    "we": "ਅਸੀਂ",
    "moved": "ਚੱਲੇ",
    "on": "ਉੱਤੇ",
    "to": "ਨੂੰ",
    "a": "ਇੱਕ",
    "brainstorming": "ਬ੍ਰੇਨਸਟਰਮਿੰਗ",
    "session": "ਸੈਸ਼ਨ",
    "concerning": "ਬਾਰੇ",
    "customer": "ਗਾਹਕ",
    "support": "ਸਹਿਯੋਗ",
    "improvements": "ਸੁਧਾਰ",
    }

# Sample English sentence
english_sentence = "after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements"

# Split the sentence into words
words = english_sentence.split()

# Translate each word using the mapping
punjabi_translation = ' '.join(english_to_punjabi_mapping.get(word, word) for word in words)

# Print the Punjabi translation
print("Punjabi Translation:", punjabi_translation)


# English to Kannada vocabulary mapping
vocabulary_mapping = {
    "so": "ಹೌದು",
    "if": "ಹೇಗಾದರೆ",
    "there": "ಅಲ್ಲಿ",
    "is": "ಇದೆ",
    "nothing": "ಯಾವುದೂ ಇಲ್ಲ",
    "else": "ಇನ್ನೊಂದು",
    "we": "ನಾವು",
    "need": "ಬೇಕು",
    "to": "ಗೆ",
    "discuss": "ಚರ್ಚಿಸಬೇಕಾಗಿದೆ",
    "let's": "ನೋಡು",
    "move": "ಹೆಜ್ಜೆ",
    "on": "ಮೇಲೆ",
    "today's": "ಇಂದಿನ",
    "agenda": "ಕಾರ್ಯಾಚರಣೆ",
    "the": "ಅದು",
    "meeting": "ಭೇಟಿ",
    "was": "ಆಗಿತ್ತು",
    "declared": "ಘೋಷಿಸಲಾಗಿತ್ತು",
    "closed": "ಮುಚ್ಚಲಾಗಿತ್ತು",
    "at": "ನಲ್ಲಿ",
    "hi": "ನಮಸ್ಕಾರ",
    "it's": "ಇದು",
    "me": "ನಾನು",
    "let": "ಬಿಟ್ಟು",
    "just": "ಸರಿ",
    "summarize": "ಸಾರಿಸು",
    "main": "ಮುಖ್ಯ",
    "points": "ಅಂಶಗಳು",
    "of": "ನ",
    "last": "ಕೊನೆಯ",
    "report": "ವರದಿ",
    "thank": "ಧನ್ಯವಾದ",
    "you": "ನೀವು",
    "tom": "ಟಾಮ್",
    "have": "ಹೊಂದಿದ್ದೇನೆ",
    "all": "ಎಲ್ಲರೂ",
    "received": "ಸ್ವೀಕರಿಸಿದ",
    "a": "ಒಂದು",
    "copy": "ನಕಲಿ",
    "of": "ನ",
    "today's": "ಇಂದಿನ",
    "you": "ನೀವು",
    "should": "ಬೇಕು",
    "sign": "ಸಹಿ",
    "up": "ಮೇಲೇ",
    "for": "ಗೆ",
    "that": "ಅದು",
    "seminar": "ಸಮಿನಾರ್",
    "next": "ಮುಂದಿನ",
    "year": "ವರ್ಷ",
    "after": "ನಂತರ",
    "briefly": "ಸಂಕ್ಷೇಪವಾಗಿ",
    "revising": "ಪುನರ್ವಿಮರ್ಶೆ",
    "the": "ಅದು",
    "changes": "ಬದಲಾವಣೆಗಳು",
    "that": "ಅದು",
    "will": "ಸಾಧ್ಯ",
    "take": "ತೆಗೆದುಕೊಳ್ಳುತ್ತವೆ",
    "place": "ಸ್ಥಳ",
    "we": "ನಾವು",
    "moved": "ಹಾರಾಡಿದ್ದೆವು",
    "on": "ಮೇಲೆ",
    "to": "ಗೆ",
    "a": "ಒಂದು",
    "brainstorming": "ಭಾವನಾ ವೃತ್ತಿ",
    "session": "ಸೆಷನ್",
    "concerning": "ಪ್ರಸಂಗದ",
    "after": "ನಂತರ",
    "customer": "ಗ್ರಾಹಕ",
    "support": "ಬೆಂಬಲ",
    "improvements": "ಮೆಚ್ಚಿಸುವಂತೆ",
    "before": "ಹಿಂದೆ",
    "i": "ನಾನು",
    "begin": "ಪ್ರಾರಂಭಿಸುತ್ತಿದ್ದೇನೆ",
    "the": "ಅದು",
    "report": "ವರದಿ",
    "i": "ನಾನು",
    "like": "ಇಷ್ಟು",
    "to": "ಗೆ",
    "get": "ಪಡೆಯಲು",
    "some": "ಕೆಲವು",
    "ideas": "ಕಲ್ಪನೆಗಳು",
    "from": "ಇಂದ",
    "you": "ನೀವು",
    "all": "ಎಲ್ಲರೂ",
    "unknown": "ಅಜ್ಞಾತ",
    "i": "ನಾನು",
    "think": "ಭಾವಿಸುತ್ತೇನೆ",
    "rural": "ಗ್ರಾಮೀಣ",
    "customers": "ಗ್ರಾಹಕರು",
    "want": "ಬಯಸುತ್ತಾರೆ",
    "to": "ಗೆ",
    "feel": "ಅನುಭವಿಸು",
    "as": "ಅಂತಹ",
    "important": "ಮುಖ್ಯ",
    "our": "ನಮ್ಮ",
    "living": "ಬದುಕು",
    "in": "ಇಲ್ಲ",
    "cities": "ನಗರಗಳು",
    "well": "ಚೆನ್ನಾಗಿ",
    "me": "ನನಗೆ",
    "with": "ಜೊತೆ",
    "this": "ಈ",
    "powerpoint": "ಪವರ್‌ಪಾಯಿಂಟ್",
    "presentation": "ಪ್ರದರ್ಶನ",
    "jack": "ಜ್ಯಾಕ್",
    "presents": "ಪ್ರದರ್ಶಿಸುತ್ತಾನೆ",
    "his": "ಅವನ",
    "how": "ಹೇಗೆ",
    "do": "ಮಾಡು",
    "feel": "ಭಾವಿಸು",
    "about": "ಬಗ್ಗೆ",
    "rural": "ಗ್ರಾಮೀಣ",
    "sales": "ಮಾರಾಟದ",
    "in": "ಇಲ್ಲ",
    "your": "ನಿಮ್ಮ",
    "districts": "ಜಿಲ್ಲೆಗಳು",
    "excuse": "ಕ್ಷಮಿಸಿ",
    "me": "ನನಗೆ",
    "i": "ನಾನು",
    "didn't": "ಇಲ್ಲಿರಲಿಲ್ಲ",
    "catch": "ಹಿಡಿಯಲು",
    "that": "ಅದು",
    "can": "ಸಾಧ್ಯ",
    "we": "ನಾವು",
    "fix": "ಮಾಡು",
    "the": "ಅದು",
    "next": "ಮುಂದಿನ",
    "meeting": "ಭೇಟಿ",
    "please": "ದಯವಿಟ್ಟು",
    "i": "ನಾನು",
    "suggest": "ಸೂಚಿಸು",
    "we": "ನಾವು",
    "break": "ವಿರಾಮ",
    "up": "ಮೇಲೇ",
    "into": "ಗೆ",
    "groups": "ಗುಂಪುಗಳು",
    "and": "ಮತ್ತು",
    "ideas": "ಕಲ್ಪನೆಗಳು",
    "we've": "ನಮಗೆ",
    "seen": "ನೋಡಿದ್ದೇವೆ",
    "presented": "ಪ್ರಸ್ತುತಿಸಲಾಗಿದೆ",
    "we": "ನಾವು",
    "are": "ಇರುವುದು",
    "considering": "ಪರಿಗಣಿಸುತ್ತಿದ್ದೇವೆ",
    "specific": "ನಿರ್ದಿಷ್ಟ",
    "data": "ಡೇಟಾ",
    "mining": "ಖನಿ",
    "procedures": "ನಿಯಮಗಳು",
    "to": "ಗೆ",
    "help": "ಸಹಾಯ",
    "deepen": "ಆಳವಾಗಿಸಲು",
    "our": "ನಮ್ಮ",
    "understanding": "ಅರಿವು",
    "i'd": "ನಾನು",
    "like": "ಇಷ್ಟು",
    "to": "ಗೆ",
    "thank": "ಧನ್ಯವಾದ",
    "jack": "ಜ್ಯಾಕ್",
    "for": "ಗೆ",
    "coming": "ಬರುವುದು",
    "our": "ನಮ್ಮ",
    "meeting": "ಭೇಟಿ",
    "today": "ಇಂದ",
    "and": "ಮತ್ತು",
    "i": "ನಾನು",
    "have": "ಹೊಂದಿದ್ದೇನೆ",
    "to": "ಗೆ",
    "a": "ಒಂದು",
    "thought": "ಭಾವನೆ",
    "let's": "ನೋಡು",
    "start": "ಪ್ರಾರಂಭಿಸು",
    "going": "ಹೋಗುವುದು",
    "around": "ಸುತ್ತ",
    "the": "ಅದು",
    "table": "ಟೇಬಲ್",
    "gathering": "ಸಮೂಹ",
    "your": "ನಿಮ್ಮ",
    "feedback": "ಪ್ರತಿಸ್ಪಂದನೆ",
    "on": "ಮೇಲೆ",
    "last": "ಕೊನೆಯ",
    "week's": "ವಾರದ",
    "presentation": "ಪ್ರದರ್ಶನ",
    "do": "ಮಾಡು",
    "you": "ನೀವು",
    "all": "ಎಲ್ಲರೂ",
    "have": "ಹೊಂದಿದ್ದೇನೆ",
    "any": "ಯಾವುದೇ",
    "other": "ಬೇರೆ",
    "questions": "ಪ್ರಶ್ನೆಗಳು",
    "before": "ಹಿಂದೆ",
    "we": "ನಾವು",
    "move": "ಹೆಜ್ಜೆ",
    "on": "ಮೇಲೆ",
    "to": "ಗೆ",
    "our": "ನಮ್ಮ",
    "next": "ಮುಂದಿನ",
    "topic": "ವಿಷಯ",
    "this": "ಈ",
    "is": "ಇದೆ",
    "important": "ಮುಖ್ಯ",
    "i": "ನಾನು",
    "think": "ಭಾವಿಸುತ್ತೇನೆ",
    "your": "ನಿಮ್ಮ",
    "feedback": "ಪ್ರತಿಸ್ಪಂದನೆ",
    "about": "ಬಗ್ಗೆ",
    "this": "ಈ",
    "matter": "ವಿಷಯ",
    "would": "ಇರಬಹುದು",
    "be": "ಆಗಬಹುದು",
    "appreciated": "ಮೆಚ್ಚಿದ",
    "what": "ಏನು",
    "are": "ಇರುವುದು",
    "your": "ನಿಮ್ಮ",
    "initial": "ಮೊದಲಿನ",
    "thoughts": "ಭಾವನೆಗಳು",
    "on": "ಮೇಲೆ",
    "the": "ಅದು",
    "proposal": "ಸೂಚನೆ",
    "i": "ನಾನು",
    "look": "ನೋಡು",
    "forward": "ಮುನ್ನಡೆ",
    "to": "ಗೆ",
    "hearing": "ಕೇಳುತ್ತಿದ್ದೇನೆ",
    "from": "ಇಂದ",
    "you": "ನೀವು",
    "all": "ಎಲ್ಲರೂ"
}

# Function to translate English sentence to Kannada
def translate_to_kannada(sentence):
    words = sentence.split()
    translated_sentence = ' '.join([vocabulary_mapping.get(word.lower(), word) for word in words])
    return translated_sentence

# Example sentence to translate
english_sentence = "after briefly revising the changes that will take place we moved on to a brainstorming session concerning after customer support improvements"

# Translating the English sentence to Kannada
kannada_translation = translate_to_kannada(english_sentence)
print("English Sentence:", english_sentence)
print("Kannada Translation:", kannada_translation)


from sklearn.metrics import classification_report

# Actual and predicted labels
actual_labels = ['Telugu', 'Telugu', 'Punjabi', 'Kannada', 'Telugu', 'Punjabi', 'Punjabi', 'Kannada', 'Kannada', 'Kannada']
predicted_labels = ['Telugu', 'Kannada', 'Punjabi', 'Kannada', 'Telugu', 'Punjabi', 'Kannada', 'Kannada', 'Kannada', 'Punjabi']

# Generate classification report
report = classification_report(actual_labels, predicted_labels)

# Print the classification report
print(report)


