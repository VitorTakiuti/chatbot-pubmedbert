
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load PubMedBERT
def load_pubmedbert():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    return tokenizer, model

# Load and preprocess JSON data
def load_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)
    return articles

def preprocess_articles(articles, tokenizer, model, device="cpu", batch_size=32):
    embeddings = []
    texts = [f"{article['Title']} {article['Abstract']}" for article in articles]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # CLS token
            embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)

# Recommend articles
def recommend_articles(query, articles, embeddings, tokenizer, model, device="cpu", top_k=5):
    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        query_embedding = model(**query_inputs).last_hidden_state[:, 0, :].cpu()  # CLS token
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    recommendations = [articles[i] for i in top_indices]
    return recommendations

# Gradio chatbot interface
def chatbot_interface(query):
    if len(query.strip()) < 3:
        return "Please provide a more detailed query to get accurate recommendations."
    recommendations = recommend_articles(query, articles, embeddings, tokenizer, model, device=device)
    response = "Here are the most relevant articles based on your query:\n\n"
    for i, rec in enumerate(recommendations, 1):
        response += (
            f"**{i}. Title**: {rec['Title']}\n"
            f"**DOI**: {rec['DOI']}\n"
            f"**Abstract**: {rec['Abstract'][:500]}...\n"
            f"**Journal**: {rec['Journal']}\n"
            f"**Publication Date**: {rec['PublicationDate']}\n\n"
        )
    return response

# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and data
    tokenizer, model = load_pubmedbert()
    model = model.to(device)
    articles = load_articles("merged_articles_fixed.json")

    # Generate embeddings
    embeddings_path = "Newembeddings.pt"
    if not os.path.exists(embeddings_path):
      print("Generating embeddings...")
      embeddings = preprocess_articles(articles, tokenizer, model, device=device)
      torch.save(embeddings, embeddings_path)
      print(f"Embeddings saved to {embeddings_path}")
    else:
      print("Loading precomputed embeddings...")
      embeddings = torch.load(embeddings_path, map_location="cpu")

    # Launch Gradio chatbot
    print("Launching chatbot...")
    gr.Interface(
        fn=chatbot_interface,
        inputs="text",
        outputs="text",
        title="Article Recommendation Chatbot",
        description="Ask about oncology, digital health, or artificial intelligence articles from PubMed.",
    ).launch()
