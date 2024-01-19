# Databricks notebook source
# MAGIC %pip install openpyxl nltk

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score, roc_auc_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel

# COMMAND ----------

df_raw = pd.read_excel('/dbfs/mnt/DAP/data/DigitalDevelopmentOperations/Documents/FY14 lending investments PDF URLs - DD Lead GP, DE4A, and Digital Tags.xlsx')
df_raw['name_objective_abstract'] = df_raw['Project Name'].fillna('') + ' ' \
    + df_raw['Development Objective Description'].fillna('') + ' ' \
    + df_raw['ABSTRACT_TEXT'].fillna('')
df = df_raw.dropna(subset=['ABSTRACT_TEXT'])
df = df[(df.FY >= 2018) & (df.FY <= 2023)] # before 2018 there is no "DE4A Manuel" = 0, after 2023 there are DD Lead GP = 1 while DE4A Manual = 0: df[(df['DD Lead GP'] != df['DE4A Manual']) & (df['DD Lead GP'] == 1)]
df

# COMMAND ----------

df.groupby('DE4A Manual').count()['Project Id']

# COMMAND ----------

df.groupby(['DE4A Manual', 'FY']).count()['Project Id']

# COMMAND ----------

balanced_accuracy_score(df['DE4A Manual'], df['Digital Tag'])

# COMMAND ----------

df[(df['DD Lead GP'] != df['DE4A Manual'])]

# COMMAND ----------

# Model specification
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
classifier = LogisticRegression(solver="liblinear", class_weight='balanced', C=0.001, random_state=42)
model = make_pipeline(vectorizer, classifier)

# Preprocessing: Shuffling the data & split train test
df = shuffle(df, random_state=42)

X = df['name_objective_abstract']
y = df['DE4A Manual']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validate with balanced accuracy
balanced_accuracy = make_scorer(balanced_accuracy_score)
recall = make_scorer(recall_score)
cv_scores_accuracy = cross_val_score(model, X_train, y_train, cv=8, scoring=balanced_accuracy)
cv_scores_recall = cross_val_score(model, X_train, y_train, cv=8, scoring=recall)
print("Cross-validated Balanced Accuracy: {:.2f}".format(cv_scores_accuracy.mean()),
      "Sensitivity: {:.2f}".format(cv_scores_recall.mean()))

# Train the model on the full training set, predict then evaluate
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
test_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
print("Test Balanced Accuracy: {:.2f}".format(test_balanced_accuracy),
    "Sensitivity: {:.2f}".format(test_recall))

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'tfidfvectorizer__max_features': [100, 500, 1000],
    # 'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    # 'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    # 'multinomialnb__alpha': [0.1, 0.5, 1.0],
}
grid_search = GridSearchCV(model, param_grid, cv=8, scoring=balanced_accuracy)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters & accuracy score
print("Best Parameters:", grid_search.best_params_)
print("Best Balanced Accuracy:", grid_search.best_score_)

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, balanced_accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import numpy as np
import nltk
nltk.download('punkt')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stemming=False):
        self.stemming = stemming
        self.ps = SnowballStemmer("english")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        preprocessed_text = X.apply(self._preprocess_text)
        return preprocessed_text

    def _preprocess_text(self, text):
        text = text.lower()
        words = word_tokenize(text)
        # Handling punctuation
        words = [word for word in words if word.isalnum()]
        if self.stemming:
            words = [self.ps.stem(word) for word in words]
        return ' '.join(words)

preprocessor = TextPreprocessor(stemming=True)
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
classifier = LogisticRegression(solver="liblinear", class_weight='balanced', C=0.001)

model = make_pipeline(
    preprocessor,
    vectorizer,
    classifier
)

# Preprocessing: Shuffling the data & split train test
df = shuffle(df, random_state=42)

X = df['name_objective_abstract']
y = df['DE4A Manual']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cross-validate with balanced accuracy
balanced_accuracy = make_scorer(balanced_accuracy_score)
recall = make_scorer(recall_score)
cv_scores_accuracy = cross_val_score(model, X_train, y_train, cv=8, scoring=balanced_accuracy)
cv_scores_recall = cross_val_score(model, X_train, y_train, cv=8, scoring=recall)
print("Cross-validated Balanced Accuracy: {:.2f}".format(cv_scores_accuracy.mean()),
      "Sensitivity: {:.2f}".format(cv_scores_recall.mean()))

# Train the model on the full training set, predict then evaluate
model.fit(X_train, y_train)

test_predictions = model.predict(X_test)
test_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
print("Test Balanced Accuracy: {:.2f}".format(test_balanced_accuracy),
      "Sensitivity: {:.2f}".format(test_recall))


# COMMAND ----------

# Experiemented with adding feature selection after TFIDF didn't improve performance
# Experiemented with adding preprocessing (lowercasing, punctuation handling & stemming) also didn't improve performance
# Experiemented MuntinominalNB instead of LogisticRegression also didn't improve performance
# pipeline = TFIDF + a simple classifer tops at Cross-validated Balanced Accuracy: 0.86 Sensitivity: 0.86

# COMMAND ----------

# TODO: try Doc2Vec: split test & train. For train, each doc has its own label, build model. Then for each train doc, run the model and rank it against all other documents, check if we can grap top n or do say 2 out of top 3 most similar have digital flagged 1 then predict digital, check accuracy to determine n (& rules). Run on test set.
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
