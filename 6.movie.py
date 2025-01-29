import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("D:\\23801928\\movdata.csv")

data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Preprocess the data
data['review'] = data['review'].str.replace('[^a-zA-Z]', ' ').str.lower()

# Split the data
X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, zero_division=0))  # Adjust zero_division as needed

def predict_review(review):
    # Preprocess the review
    review_cleaned = review.replace('[^a-zA-Z]', ' ').lower()
    # Vectorize the review
    review_vectorized = vectorizer.transform([review_cleaned])
    # Predict the sentiment
    prediction = model.predict(review_vectorized)
    return "Positive" if prediction[0] == 1 else "Negative"

# Example usage
new_review = input("Enter your Review: ")
print(predict_review(new_review))
