# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

df_raw = pd.read_excel('/dbfs/mnt/DAP/data/DigitalDevelopmentOperations/Documents/FY14 lending investments PDF URLs - DD Lead GP, DE4A, and Digital Tags.xlsx')
df_raw

# COMMAND ----------

project_df = spark.table("digital.ops_projects").toPandas()
project_df['name_objective_abstract'] = project_df['proj_name'].fillna('') + ' ' \
    + project_df['dev_objective_desc'].fillna('') + ' ' \
    + project_df['abstract_text'].fillna('')
project_df

# COMMAND ----------

from pathlib import Path
import joblib

MODEL_PATH = '/dbfs/mnt/DAP/data/DigitalDevelopmentOperations/models/tfidf_logit.pkl'
model = joblib.load(MODEL_PATH)
model

# COMMAND ----------

project_df['predicted_digital'] = model.predict(project_df.name_objective_abstract)
project_df

# COMMAND ----------

merged = pd.merge(project_df, df_raw, left_on='proj_id', right_on='Project Id', how='left')
merged['is_digital'] = np.where(merged['DE4A Manual'].notnull(), merged['DE4A Manual'], merged['predicted_digital'])
merged = merged.astype({'is_digital': 'int'})
merged

# COMMAND ----------

sdf = spark.createDataFrame(merged[['proj_id', 'is_digital']])
sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("digital.ops_projects_predicted")

# COMMAND ----------

## TODO: Use this once UC is fixed
# project_df = spark.sql("""
#     SELECT p.proj_id, proj_name, 
#         dev_objective_desc, 
#         proj_apprvl_fy,
#         proj_stat_name,
#         cntry_short_name,
#         -- cntry_long_name,
#         abstract_text,
#         doc_type_short_name
#     FROM prd_corpdata.dm_reference_gold.dim_project p
#         JOIN (
#             SELECT d.proj_id, d.abstract_text, d.doc_type_short_name
#             FROM prd_corpdata.dm_reference_gold.v_dim_imagebank_document d
#             JOIN (
#                 SELECT proj_id, MAX(node_id) AS max_node_id
#                 FROM prd_corpdata.dm_reference_gold.v_dim_imagebank_document
#                 WHERE doc_type_short_name IN ('PAD', 'PGD', 'PID', 'PJPR') 
#                     AND lang_name = 'English' 
#                     AND abstract_text IS NOT NULL
#                 GROUP BY proj_id
#             ) AS dn
#             ON d.proj_id = dn.proj_id
#             AND d.node_id = dn.max_node_id
#         ) as proj_doc
#     ON p.proj_id = proj_doc.proj_id
#     WHERE proj_stat_name != 'Dropped' 
#     AND proj_stat_name != 'Canceled'
#     AND proj_stat_name != 'Draft'
#     ORDER BY proj_apprvl_fy DESC
# """)
# project_df
