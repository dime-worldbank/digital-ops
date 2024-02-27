# Databricks notebook source
# MAGIC %pip install gensim openpyxl

# COMMAND ----------

# MAGIC %md
# MAGIC ### Doc2Vec model 1: 
# MAGIC Split test & train. For train set, each doc has its own label, build model. Feed the resulting vectors to logistic classifer with the binary digital class as the target variable.
# MAGIC Reference: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py

# COMMAND ----------

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
import pandas as pd
import gensim

# COMMAND ----------

df_raw = pd.read_excel('/Volumes/prd_dap/volumes/dap/data/DigitalDevelopmentOperations/Documents/FY14 lending investments PDF URLs - DD Lead GP, DE4A, and Digital Tags.xlsx')
df_raw['name_objective_abstract'] = df_raw['Project Name'].fillna('') + ' ' \
    + df_raw['Development Objective Description'].fillna('') + ' ' \
    + df_raw['ABSTRACT_TEXT'].fillna('')
df_cleaned = df_raw.dropna(subset=['ABSTRACT_TEXT'])
df_cleaned = df_cleaned[(df_cleaned.FY >= 2018) & (df_cleaned.FY <= 2023)] # before 2018 there is no "DE4A Manuel" = 0, after 2023 there are DD Lead GP = 1 while DE4A Manual = 0: df[(df['DD Lead GP'] != df['DE4A Manual']) & (df['DD Lead GP'] == 1)]
df_cleaned

# COMMAND ----------

projects_sdf_with_gp = spark.sql("""
    SELECT proj_id, lead_gp_name
    FROM prd_corpdata.dm_reference_gold.dim_project
""")
projects_with_gp = projects_sdf_with_gp.toPandas()
projects_with_gp

# COMMAND ----------

df_merged = pd.merge(df_cleaned, projects_with_gp, left_on='Project Id', right_on='proj_id', how='left')
df = df_merged[df_merged.lead_gp_name.isin(['Digital Development', 'Education', 'Finance, Competitiveness and Innovation', 'Governance', 'Social Protection & Jobs'])]
df = df.drop(columns=['proj_id'])
df

# COMMAND ----------

df['tokenized_text'] = df['name_objective_abstract'].apply(gensim.utils.simple_preprocess)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# only training set needs tagged documents with unique identifiers
train_tagged = [TaggedDocument(words=row['tokenized_text'], tags=[index]) for index, row in train_df.iterrows()]
test_untagged = test_df.tokenized_text

# Build and train Doc2Vec model
model = Doc2Vec(vector_size=300, min_count=2, epochs=20)
model.build_vocab(train_tagged)

# COMMAND ----------

model.wv.get_vecattr('digital', 'count')

# COMMAND ----------

model.train(train_tagged, total_examples=model.corpus_count, epochs=model.epochs)

# COMMAND ----------

import collections

ranks = []
second_ranks = []
for doc in train_tagged:
    inferred_vector = model.infer_vector(doc.words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc.tags[0])
    ranks.append(rank)
    second_ranks.append(sims[1])

counter = collections.Counter(ranks)
# print(counter) # Sanity check: expect most of the ranks should be 0, meaning the document is found most similar to itself
similar_to_self_share = counter[0]/sum(counter.values())
assert similar_to_self_share > 0.99, f"Expect most of the documents to be most similar to itself but got {similar_to_self_share}, {counter}"
counter

# COMMAND ----------

from sklearn.metrics import make_scorer, balanced_accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

# Generate document vectors for training and test sets
X_train_doc2vec = [model.dv[index] for index in train_df.index]
X_test_doc2vec = [model.infer_vector(tokens) for tokens in test_untagged]

# Extract labels
y_train = train_df['DE4A Manual']
y_test = test_df['DE4A Manual']

# Build a pipeline with feature extraction and classifier
classifier = LogisticRegression(solver='liblinear', class_weight='balanced')
model_pipeline = make_pipeline(classifier)

# Cross-validate with balanced accuracy
balanced_accuracy = make_scorer(balanced_accuracy_score)
recall = make_scorer(recall_score)
cv_scores_accuracy = cross_val_score(model_pipeline, X_train_doc2vec, y_train, cv=8, scoring=balanced_accuracy)
cv_scores_recall = cross_val_score(model_pipeline, X_train_doc2vec, y_train, cv=8, scoring=recall)
print("Cross-validated Balanced Accuracy: {:.2f}".format(cv_scores_accuracy.mean()),
      "Sensitivity: {:.2f}".format(cv_scores_recall.mean()))

# Train the model on the full training set, predict then evaluate
model_pipeline.fit(X_train_doc2vec, y_train)

test_predictions = model_pipeline.predict(X_test_doc2vec)

test_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
print("Test Balanced Accuracy: {:.2f}".format(test_balanced_accuracy),
    "Sensitivity: {:.2f}".format(test_recall))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Doc2vec model 2
# MAGIC Binary digital tag for each document then directly use the model on test set

# COMMAND ----------

# only training set needs tagged documents with digital flag as tags
train_tagged = [TaggedDocument(words=row['tokenized_text'], tags=[row['DE4A Manual']]) for i, row in train_df.iterrows()]
test_untagged = test_df.tokenized_text

# Build and train Doc2Vec model
model = Doc2Vec(vector_size=300, min_count=2, epochs=20)
model.build_vocab(train_tagged)

model.train(train_tagged, total_examples=model.corpus_count, epochs=model.epochs)

# COMMAND ----------

ranks = []
for doc in train_tagged:
    inferred_vector = model.infer_vector(doc.words)
    sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    rank = [docid for docid, sim in sims].index(doc.tags[0])
    ranks.append(rank)

counter = collections.Counter(ranks)
similar_to_self_share = counter[0]/sum(counter.values())
assert similar_to_self_share > 0.95, f"Expect most of the documents to be most similar to its own digital tag but got {similar_to_self_share}, {counter}"
similar_to_self_share

# COMMAND ----------

test_pred = []
for tokens in test_untagged:
    vectors = model.infer_vector(tokens)
    sims = model.dv.most_similar([vectors], topn=1)
    test_pred.append(sims[0][0])

test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
print("Test Balanced Accuracy: {:.2f}".format(test_balanced_accuracy),
    "Sensitivity: {:.2f}".format(test_recall))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC Doc2Vec model 1 has balanced accuracy ~76% while model 2 ~74% with very low sensitivity. Note that both models are not well tuned so there might be potential for improvements.
