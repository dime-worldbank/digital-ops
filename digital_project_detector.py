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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the labeled dataset
# MAGIC
# MAGIC From Clement:
# MAGIC > - For example, one could consider that the list of all Digital Development GP could be a good place to start to define “digital projects”. This is the narrowest definition we have.
# MAGIC > - Alternatively, we have used a keywords-based approach to identify digital projects (the so-called “digital tag”). This is the broadest definition.
# MAGIC > - In the middle, we have a manually collected dataset based on the Digital Economy for Africa initiative (the “DE4A tag”). This is the most accurate but it was collected only for five GPs between FY 2018 and 2023.
# MAGIC
# MAGIC Notes from exploratory analysis: 
# MAGIC - Use DE4A as the ground truth since it is the most accurate of the 3 options.
# MAGIC - The DE4A tag does not seem to be geographically limited to Africa.
# MAGIC - The biggest problem with this labeled dataset is that the non-digital (0) class may contain digital projects. These are false negatives in the labeled dataset we are using as ground truth. Given the "garbage in, garbage out" principle, it might be worth going through all projects between 2018 and 2023 and manually review and correct the "DE4A tag" = 0 projects.

# COMMAND ----------

df_raw = pd.read_excel('/Volumes/prd_dap/volumes/dap/data/DigitalDevelopmentOperations/Documents/FY14 lending investments PDF URLs - DD Lead GP, DE4A, and Digital Tags.xlsx')
df_raw['name_objective_abstract'] = df_raw['Project Name'].fillna('') + ' ' \
    + df_raw['Development Objective Description'].fillna('') + ' ' \
    + df_raw['ABSTRACT_TEXT'].fillna('')
df = df_raw[(df_raw.FY >= 2018) & (df_raw.FY <= 2023)] # before 2018 there is no "DE4A Manuel" = 0, after 2023 there are DD Lead GP = 1 while DE4A Manual = 0: df[(df['DD Lead GP'] != df['DE4A Manual']) & (df['DD Lead GP'] == 1)]
df

# COMMAND ----------

by_de4a = df.groupby('DE4A Manual').count()['Project Id']
display(by_de4a)

# COMMAND ----------

df.groupby(['DE4A Manual', 'FY']).count()['Project Id']

# COMMAND ----------

balanced_accuracy_score(df['DE4A Manual'], df['Digital Tag'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Text preprocessing

# COMMAND ----------

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model specification & training

# COMMAND ----------

# Model specification
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
classifier = LogisticRegression(solver="liblinear", class_weight='balanced', C=0.001, random_state=42)
# classifier = MultinomialNB(alpha=0.1)
model = make_pipeline(preprocessor, vectorizer, classifier)

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
    # 'tfidfvectorizer__max_features': [500, 1000],
    'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
    # 'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__C': [0.001, 0.01, 0.1, 1.0],
    # 'multinomialnb__alpha': [0.1, 0.5, 1.0],
}
grid_search = GridSearchCV(model, param_grid, cv=8, scoring=balanced_accuracy)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters & accuracy score
print("Best Parameters:", grid_search.best_params_)
print("Best Balanced Accuracy:", grid_search.best_score_)

# COMMAND ----------

# Experiemented with adding feature selection after TFIDF didn't improve performance
# Experiemented with adding preprocessing (lowercasing, punctuation handling & stemming) also didn't improve performance
# Experiemented MuntinominalNB instead of LogisticRegression also didn't improve performance
# pipeline = TFIDF + a simple classifer tops at Cross-validated Balanced Accuracy: 0.86 Sensitivity: 0.86

# COMMAND ----------

from pathlib import Path
import joblib

MODEL_DIR = '/Volumes/prd_dap/volumes/dap/data/DigitalDevelopmentOperations/models'
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

joblib.dump(model, f'{MODEL_DIR}/tfidf_logit.pkl', compress = 1)
