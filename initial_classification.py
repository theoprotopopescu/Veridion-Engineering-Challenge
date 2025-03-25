import pandas as pd
import numpy as np
import torch
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

# Load the taxonomy file (adjust the sheet name and column names as needed)
taxonomy_df = pd.read_excel('insurance_taxonomy.xlsx')
# Assume taxonomy_df has a column named "Label" that contains the 220 possible categories
taxonomy_labels = taxonomy_df['label'].dropna().tolist()

# Load the company data
company_df = pd.read_csv('ml_insurance_challenge.csv')
# Assume company_df has columns 'Company Description' and 'Business Tags'
# You can also include other columns if needed (e.g., Sector, Category, Niche)
company_df['combined_text'] = company_df.apply(
    lambda row: f"{row['description'] if pd.notnull(row['description']) else ''} " \
                f"{row['business_tags'] if pd.notnull(row['business_tags']) else ''} " \
                f"{row['sector'] if pd.notnull(row['sector']) else ''} " \
                f"{row['category'] if pd.notnull(row['category']) else ''} " \
                f"{row['niche'] if pd.notnull(row['niche']) else ''}",
    axis=1
)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

company_df["cleaned_text"] = company_df["combined_text"].apply(clean_text)

print(company_df)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for the taxonomy labels
# (If your taxonomy file had more context than just the label name, you could include that as well)
taxonomy_embeddings = model.encode(taxonomy_labels, convert_to_tensor=True)

# Compute embeddings for each company’s combined text
company_embeddings = model.encode(company_df['cleaned_text'].tolist(), convert_to_tensor=True)

# Set a similarity threshold; you might need to tune this
similarity_threshold = 0.5

# For each company, compute cosine similarities with all taxonomy labels
assigned_labels = []
for company_emb in company_embeddings:
    # Compute cosine similarity between the company and all taxonomy label embeddings
    cosine_scores = util.cos_sim(company_emb, taxonomy_embeddings)[0]
    # Get indices of taxonomy labels with similarity above the threshold
    indices = np.where(cosine_scores >= similarity_threshold)[0]
    # Alternatively, if you want the top k labels, you could use:
    #top_results = torch.topk(cosine_scores, k=3)
    selected = [taxonomy_labels[i] for i in indices[0:3]]
    assigned_labels.append(selected)

# Add the predicted taxonomy labels to the DataFrame
company_df['Predicted_Taxonomy_Labels'] = assigned_labels

# Save the annotated results to a new CSV file
company_df.to_csv('ml_insurance_challenge_annotated_cosine.csv', index=False)

print("Classification complete. Annotated file saved as 'ml_insurance_challenge_annotated.csv'")