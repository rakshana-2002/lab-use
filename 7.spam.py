import nltk
import pandas as pd
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy


file_path = 'spam.csv'  
df = pd.read_csv("D:\\23801928\\spam.csv")


print(df.head())

stop_words = set(stopwords.words('english'))

def extract_features(message):
    words = word_tokenize(message.lower())  
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    return {word: True for word in filtered_words}


df = df.sample(frac=1, random_state=42)  


feature_sets = [(extract_features(message), label) for message, label in zip(df['message'], df['label'])]
train_size = int(0.8 * len(feature_sets))
train_features = feature_sets[:train_size]
test_features = feature_sets[train_size:]


classifier = NaiveBayesClassifier.train(train_features)


accuracy_score = accuracy(classifier, test_features)
print(f"Accuracy: {accuracy_score * 100:.2f}%")


test_message = "india won the match yesterday"
test_features = extract_features(test_message)
prediction = classifier.classify(test_features)
print(f"Prediction for the message: '{test_message}' is {prediction}")
