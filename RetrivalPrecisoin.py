# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 02:47:58 2026

@author: User
"""

import pandas as pd

# -------------------------
# Parameters
# -------------------------
excel_file = "D:\\Research\\GEN-AI\\PH2\\history\\arabic\\old\\llama3.1-Arabic_results_updated.xlsx"  # <-- replace with your file name
k = 4  # top-k for Recall@k and MRR@k

# -------------------------
# Load Excel file
# -------------------------
df = pd.read_excel(excel_file)

# Lists to store per-question metrics
recall_list = []
mrr_list = []

# -------------------------
# Compute Recall@k and MRR@k
# -------------------------
for idx, row in df.iterrows():
    # Split the retrieved chunks by 'passage:'
    source_text = str(row['source_documents'])
    chunks = source_text.split("passage:")
    chunks = [c.strip() for c in chunks if c.strip()]  # remove empty strings
    
    # Get the ground truth answer
    ground_truth = str(row['reference_answer']).strip()
    
    # Initialize rank as None
    rank = None
    
    # Search through chunks
    for i, chunk in enumerate(chunks, start=1):
        if ground_truth.lower() in chunk.lower():  # case-insensitive match
            rank = i
            break
    
    # Compute Recall@k
    recall = 1 if rank is not None and rank <= k else 0
    recall_list.append(recall)
    
    # Compute MRR@k
    mrr = 1/rank if rank is not None and rank <= k else 0
    mrr_list.append(mrr)

# -------------------------
# Compute overall metrics
# -------------------------
overall_recall_at_k = sum(recall_list) / len(recall_list)
overall_mrr_at_k = sum(mrr_list) / len(mrr_list)

print(f"Recall@{k}: {overall_recall_at_k:.4f}")
print(f"MRR@{k}: {overall_mrr_at_k:.4f}")

# -------------------------
# Optional: add new columns to dataframe
# -------------------------
df[f"recall_at_{k}"] = recall_list
df[f"MRR_at_{k}"] = mrr_list

# Save updated Excel with metrics
df.to_excel("qa_with_retrieval_metrics.xlsx", index=False)
print("Saved Excel with retrieval metrics.")
