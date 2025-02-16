import torch

embeddings_path = "Newembeddings.pt"

# Carregar o arquivo sem ocupar muita memória
try:
    with open(embeddings_path, "rb") as f:
        embeddings = torch.load(f, map_location="cpu")
    print(f"Carregado com sucesso! Tipo de dados: {type(embeddings)}")
except Exception as e:
    print(f"Erro ao carregar o arquivo: {e}")

# Verificando as dimensões do tensor de embeddings
print(f"Dimensões do tensor de embeddings: {embeddings.shape}")
print(f"Tipo de dados dos embeddings: {embeddings.dtype}")
print(f"Exemplo dos primeiros valores:\n {embeddings[0][:5]}")  # Exibe os primeiros 5 valores do primeiro embedding

from sklearn.decomposition import PCA
import torch



# Converter embeddings para float16 para economizar memória
embeddings = embeddings.half()

# Salvar os embeddings quantizados
torch.save(embeddings, "embeddings_quantized.pt")

print("✅ Embeddings quantizados e salvos com sucesso!")

print(f"Dimensão dos embeddings: {embeddings.shape}")
print(f"Tipo de dados: {embeddings.dtype}")
print(f"Primeiro embedding: {embeddings[0][:10]}")
print(f"Maior valor: {embeddings.max()}, Menor valor: {embeddings.min()}")

from sklearn.metrics.pairwise import cosine_similarity

# Pegando os dois primeiros embeddings para teste
embedding_1 = embeddings[0].reshape(1, -1)
embedding_2 = embeddings[1].reshape(1, -1)

# Calculando a similaridade entre os dois primeiros embeddings
similarity = cosine_similarity(embedding_1, embedding_2)[0][0]
print(f"Similaridade entre os dois primeiros artigos: {similarity:.4f}")

# Supondo que os títulos dos artigos estejam na lista `article_titles`
article_index = 0  # Testando o primeiro artigo

print(f"🔹 Artigo {article_index}: {article_titles[article_index]}")
print(f"🧠 Embedding correspondente:\n {embeddings[article_index][:10]}")

query = "AI in health"  # Teste com uma query real

query_inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
with torch.no_grad():
    query_embedding = model(**query_inputs).last_hidden_state[:, 0, :].cpu()

print(f"🔹 Embedding gerado para a query: {query_embedding[:10]}")

print(f"🔹 Índices recomendados: {recommended_indices_list}")
print(f"🔹 Índices relevantes esperados: {relevant_indices_list}")
