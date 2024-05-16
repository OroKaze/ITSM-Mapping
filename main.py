from typing import Union

# from fastapi import FastAPI

import pandas as pd
import os
import re
import functools
import random
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
import json

# Text Processing Library
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Sklearn Vectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Membaca data dari file .txt
with open('Service-Test-NonAPI/data/data_notes.txt', 'r') as file:
    data_txt = file.read()

# Mengonversi data menjadi format JSON
data_json = json.loads(data_txt)

# Menuliskan data JSON ke file .json
with open('Service-Test-NonAPI/data/data_notes.json', 'w') as json_file:
    json.dump(data_json, json_file, indent=4)

# Mengambil nilai dari atribut "note"
notes = []
for item in data_json['result']['data']:
    notes.append(item['note'])

# Membuat DataFrame dengan Pandas
df = pd.DataFrame(notes, columns=['note'])

# Memuat model dan objek pemrosesan teks dari file
with open('Service-Test-NonAPI/models/stemmer.pkl', 'rb') as f:
    stemmer = pickle.load(f)

with open('Service-Test-NonAPI/models/cleaner_f.pkl', 'rb') as f:
    cleaner_f = pickle.load(f)

with open('Service-Test-NonAPI/models/model_LR.pkl', 'rb') as f:
    model_LR = pickle.load(f)

# -------------------------------------------------------------------------------------------------------------------------------------

# # Fungsi untuk membersihkan dan menstem teks
# def preprocess_text(text, stemmer, cleaner_f):
#     # Membersihkan teks
#     text_cleaned = ' '.join(cleaner_f.build_analyzer()(text))
#     # Stem teks
#     stemmed_text = ' '.join([stemmer.stem(word) for word in text_cleaned.split()])
#     return stemmed_text

# # Fungsi untuk memberi label pada teks menggunakan regex
# def label_text(text_clean):
#     if re.search(r'workzone', text_clean, re.IGNORECASE):
#         label = 'workzone'
#     elif re.search(r'dispatch', text_clean, re.IGNORECASE):
#         label = 'dispatch'
#     elif re.search(r'service\s+type', text_clean, re.IGNORECASE):
#         label = 'service'
#     else:
#         label = ''  # Label default jika tidak ada yang cocok

#     return text_clean, label

# # Fungsi untuk mencari ID yang mengandung kata "inc" menggunakan regex
# def find_ticket_id(text):
#     ticket_id_match = re.search(r'INC\d+', text, re.IGNORECASE)
#     if ticket_id_match:
#         ticket_id = ticket_id_match.group(0)
#     else:
#         ticket_id = ''  # Jika tidak ada yang cocok, berikan nilai default

#     return ticket_id

# # Memproses teks
# processed_text = preprocess_text(df['note'].values, stemmer, cleaner_f)

# # Memberi label pada teks
# text_labeled = label_text(processed_text)

# # Mendapatkan ID yang mengandung kata "inc"
# ticket_id = find_ticket_id(data_txt)

# # Simpan hasil proses ke dalam DataFrame
# df = pd.DataFrame({'data_bersih': [text_labeled[0]], 'label': [text_labeled[1]], 'ticket_id': [ticket_id]})

# # Simpan DataFrame ke dalam file CSV
# df.to_csv('Service-Test-NonAPI/data/hasil_proses.csv', index=False)

# --------------------------------------------------------------------------------------------------------------------------------------

# Function to preprocess text
def preprocess_text(text, stemmer, cleaner_f):
    text_cleaned = ' '.join(cleaner_f.build_analyzer()(text))
    stemmed_text = ' '.join([stemmer.stem(word) for word in text_cleaned.split()])
    return stemmed_text

# Function to label text using regex
def label_text(text_clean):
    if re.search(r'workzone', text_clean, re.IGNORECASE):
        label = 'workzone'
    elif re.search(r'dispatch | dispath', text_clean, re.IGNORECASE):
        label = 'dispatch'
    elif re.search(r'service\s+type', text_clean, re.IGNORECASE):
        label = 'service'
    else:
        label = ''  # Default label if none match

    return label

# Function to find ticket ID using regex
def find_ticket_id(text):
    ticket_id_match = re.search(r'INC\d+', text, re.IGNORECASE)
    if ticket_id_match:
        ticket_id = ticket_id_match.group(0)
    else:
        ticket_id = ''  # Default value if none match

    return ticket_id

# Preprocess text in 'note' column
df['note_processed'] = df['note'].apply(lambda x: preprocess_text(x, stemmer, cleaner_f))

# Label text and extract ticket IDs
# df['label'] = df['note_processed'].apply(label_text)

# ---------------------------------------------------------------------------------------------
# Create and train TfidfVectorizer
Tfidf_vect = TfidfVectorizer(max_features=1000)
Tfidf_vect.fit(df['note_processed'])

# Transform text data into numerical features
Test_X_Tfidf = Tfidf_vect.transform(df['note_processed'])

# Predict labels using the model
df['label'] = model_LR.predict(Test_X_Tfidf)
# ---------------------------------------------------------------------------------------------

# Get ID from data
df['ticket_id'] = df['note_processed'].apply(find_ticket_id)

# Filter to obtain rows with ticket IDs
filtered_df = df[df['ticket_id'] != '']

# Save the filtered DataFrame to CSV
filtered_df.to_csv('Service-Test-NonAPI/data/hasil_proses.csv', index=False)
