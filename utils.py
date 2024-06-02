# utils.py
import nltk
import string
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
stopword = nltk.corpus.stopwords.words('english')

# Load the model from the .pkl file
with open('classifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the vectorizer from the .pkl file
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def replace_dash_with_space(input_string):
    return input_string.replace("-", " ")

def lower(text):
    words = text.split()
    lower = [word.lower() for word in words]
    return ' '.join(lower)

def hyperlinks(text):
    pattern = r'http\S+|www\S+'
    removed = re.sub(pattern, '', text)
    return removed

def remove_large_spaces(text):
    pattern = r'\s+'
    removed_spaces = re.sub(pattern, ' ', text)
    return removed_spaces.strip()

def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stopword])
    return text

def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def remove_non_word_characters(sentence):
    pattern = r'\W+'
    cleaned_sentence = re.sub(pattern, ' ', sentence)
    return cleaned_sentence

def remove_numbers(text):
    pattern = r'\d+'
    removed_numbers = re.sub(pattern, '', text)
    return removed_numbers

def remove_html(text):
    html_re = re.compile(r'<.*?>')
    text = re.sub(html_re, '', text)
    return text

def remove_date_time(text):
    date_pattern = r"\d{1,2}/\d{1,2}/\d{2,4}"
    time_pattern = r"\d{1,2}:\d{2}([AP]M)?"
    text_without_date = re.sub(date_pattern, "", text)
    text_without_date_time = re.sub(time_pattern, "", text_without_date)
    return text_without_date_time

def remove_mentions_hashtags(text):
    text_without_mentions = re.sub(r"@\w+", "", text)
    text_without_mentions_hashtags = re.sub(r"#\w+", "", text_without_mentions)
    return text_without_mentions_hashtags

functions = [replace_dash_with_space, lower, hyperlinks, remove_large_spaces, remove_stopwords, remove_punctuation,
             remove_non_word_characters, remove_numbers, remove_html, remove_date_time,
             remove_mentions_hashtags]

def cleaned_tokenized(x):
    result = x
    for func in functions:
        result = func(result)
    
    result = nltk.word_tokenize(result)
    result = ' '.join(result)
    
    # Use the vectorizer to transform the result
    result = vectorizer.transform([result])
    result = model.predict(result)[0]

    return result
