"""
train_model.py — run this ONCE to train the model and save it to disk.
After this finishes, app.py loads the saved model instead of retraining
every time someone uses the web app.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

# --- 1. Load the data ---
# The UCI file is tab-separated with no header: label, then message
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
print(f"Loaded {len(df)} messages "
      f"({(df['label'] == 'spam').sum()} spam, {(df['label'] == 'ham').sum()} ham)")

# --- 2. Split into training and test sets ---
# The model NEVER sees the test set during training — this is what makes
# the evaluation below honest instead of the model just memorizing answers.
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# --- 3. Convert text into numbers (TF-IDF) ---
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 4. Train the model ---
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- 5. Evaluate honestly, on data it has never seen ---
predictions = model.predict(X_test_vec)
print(f"\nAccuracy:  {accuracy_score(y_test, predictions):.3f}")
print(f"Precision: {precision_score(y_test, predictions, pos_label='spam'):.3f}  "
      f"(of messages flagged spam, how many really were)")
print(f"Recall:    {recall_score(y_test, predictions, pos_label='spam'):.3f}  "
      f"(of all real spam, how much did we catch)")
print("\nConfusion matrix (rows=actual, cols=predicted, order=[ham, spam]):")
print(confusion_matrix(y_test, predictions, labels=['ham', 'spam']))

# --- 6. Save the trained model + vectorizer so app.py can load them ---
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nSaved model.pkl and vectorizer.pkl — ready to run app.py")
