# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Classification Model
# MAGIC The classifer was trained using a manually labeled project dataset. Source code [here](https://github.com/weilu/digital-ops/blob/main/digital_project_detector.py).

# COMMAND ----------

from pathlib import Path
import joblib

MODEL_PATH = '/Volumes/prd_dap/volumes/dap/data/DigitalDevelopmentOperations/models/tfidf_logit.pkl'
model = joblib.load(MODEL_PATH)
model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Labeled Project Data
# MAGIC Read input excel data file with past ops projects as rows and column "DE4A Manual" as the ground truth indicating if each project has any digital component. The source of the this file resides on SharePoint/OneDrive [here](https://worldbankgroup.sharepoint.com.mcas.ms/sites/dap/DigitalDevelopmentOperations/Shared%20Documents/Forms/AllItems.aspx). If a project is found in this source file, the prediction result will take on the value of the "DE4A Manual" column. In other words, if any project found in source is mislabeled the prediction will also be incorrect. In such cases, updating them at the source will fix the prediction. Changes to the source will take 5-10 minutes to be reflected in the data lake after which the data pipeline needs to be manually triggered for the model and prediction results in the data store to be updated.

# COMMAND ----------

df_raw = pd.read_excel('/Volumes/prd_dap/volumes/dap/data/DigitalDevelopmentOperations/Documents/FY14 lending investments PDF URLs - DD Lead GP, DE4A, and Digital Tags.xlsx')
df_raw

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Active and Pipeline Projects
# MAGIC Read up-to-date active and pipeline projects in ops workspace for prediction by the classification model.

# COMMAND ----------

project_sdf = spark.sql("""
    SELECT p.proj_id, proj_name, 
        dev_objective_desc, 
        proj_apprvl_fy,
        proj_stat_name,
        cntry_short_name,
        -- cntry_long_name,
        abstract_text,
        doc_type_short_name
    FROM prd_corpdata.dm_reference_gold.dim_project p
        JOIN (
            SELECT d.proj_id, d.abstract_text, d.doc_type_short_name
            FROM prd_corpdata.dm_reference_gold.v_dim_imagebank_document d
            JOIN (
                SELECT proj_id, MAX(node_id) AS max_node_id
                FROM prd_corpdata.dm_reference_gold.v_dim_imagebank_document
                WHERE doc_type_short_name IN ('PAD', 'PGD', 'PID', 'PJPR') 
                    AND lang_name = 'English' 
                    AND abstract_text IS NOT NULL
                GROUP BY proj_id
            ) AS dn
            ON d.proj_id = dn.proj_id
            AND d.node_id = dn.max_node_id
        ) as proj_doc
    ON p.proj_id = proj_doc.proj_id
    WHERE proj_stat_name != 'Dropped' 
    AND proj_stat_name != 'Canceled'
    AND proj_stat_name != 'Draft'
    ORDER BY proj_apprvl_fy DESC
""")
project_sdf

# COMMAND ----------

# MAGIC %md
# MAGIC Construct parameter needed for model input:

# COMMAND ----------

project_df = project_sdf.toPandas()
project_df['name_objective_abstract'] = project_df['proj_name'].fillna('') + ' ' \
    + project_df['dev_objective_desc'].fillna('') + ' ' \
    + project_df['abstract_text'].fillna('')
project_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predict and Merge with Manual Labels

# COMMAND ----------

project_df['predicted_digital'] = model.predict(project_df.name_objective_abstract)
project_df

# COMMAND ----------

merged = pd.merge(project_df, df_raw, left_on='proj_id', right_on='Project Id', how='left')
merged['is_digital'] = np.where(merged['DE4A Manual'].notnull(), merged['DE4A Manual'], merged['predicted_digital'])
merged = merged.astype({'is_digital': 'int'})
merged

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Results to Data Store

# COMMAND ----------

sdf = spark.createDataFrame(merged[['proj_id', 'is_digital']])
sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("digital.ops_projects_predicted")
