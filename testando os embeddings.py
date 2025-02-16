import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configuração do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar PubMedBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext").to(device)


import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ================================
# PARTE A - Definições e Dados
# ================================

# Exemplo de queries de teste
test_queries = [
    "AI in health",
    "cancer treatment",
    "telemedicine applications"
]

# Exemplo de índices relevantes (você deve definir manualmente ou a partir de algum critério).
# Cada posição da lista corresponde ao array de índices relevantes para aquela query.
relevant_indices_list = [
    [0, 1, 5],     # índices relevantes para a primeira query
    [10, 15],      # índices relevantes para a segunda query
    [7, 9, 12]     # índices relevantes para a terceira query
]

# Valor de k que usaremos para Precision@k
k = 5

# Observação: Presumo que você já tenha as variáveis tokenizer e model carregadas,
# algo como:
# tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
# model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
# model.to(device)

# Também presumo que você já tenha embeddings salvos em:
# embeddings = torch.load("embeddings.pt", map_location="cpu")
# Este 'embeddings' deve ser um tensor ou array com forma [num_artigos, dim_embed]

# Caso você tenha o embeddings em GPU, você pode convertê-lo para CPU se quiser:
# embeddings = embeddings.to("cpu")

# ================================
# PARTE B - Definindo as Funções de Métricas
# ================================

def precision_at_k(recommended_indices, relevant_indices, k):
    """
    Calcula Precision@k:
    A fração de documentos relevantes entre os k primeiros retornados.
    
    recommended_indices: lista ou array dos índices recomendados (ordenados por relevância decrescente).
    relevant_indices: conjunto ou lista de índices relevantes.
    k: inteiro que define até onde contamos a precisão.
    """
    if k > len(recommended_indices):
        k = len(recommended_indices)
    recommended_top_k = set(recommended_indices[:k])
    relevant_set = set(relevant_indices)
    # interseção entre top_k e relevantes
    num_relevant_in_top_k = len(recommended_top_k.intersection(relevant_set))
    return num_relevant_in_top_k / k

def mean_average_precision(recommended_indices_list, relevant_indices_list):
    """
    Calcula MAP (Mean Average Precision).
    Percorre cada consulta, soma as precisões nos pontos onde encontra
    documentos relevantes e divide pelo total de relevantes. 
    Depois faz a média entre todas as consultas.
    """
    all_ap = []
    for recommended_indices, relevant_indices in zip(recommended_indices_list, relevant_indices_list):
        ap = 0.0
        num_hits = 0.0
        relevant_set = set(relevant_indices)
        for i, rec_index in enumerate(recommended_indices):
            if rec_index in relevant_set:
                num_hits += 1
                # Precisão em i+1
                prec_i = num_hits / (i + 1)
                ap += prec_i
        if len(relevant_indices) > 0:
            ap /= len(relevant_indices)
        all_ap.append(ap)
    return sum(all_ap) / len(all_ap) if all_ap else 0.0

def mean_reciprocal_rank(recommended_indices_list, relevant_indices_list):
    """
    Calcula o MRR (Mean Reciprocal Rank).
    Para cada consulta, identifica a posição do primeiro documento relevante
    e calcula o Reciprocal Rank (1 / posição). Faz a média em todas as consultas.
    """
    mrr_sum = 0.0
    for recommended_indices, relevant_indices in zip(recommended_indices_list, relevant_indices_list):
        reciprocal_rank = 0.0
        relevant_set = set(relevant_indices)
        for i, rec_index in enumerate(recommended_indices):
            if rec_index in relevant_set:
                reciprocal_rank = 1.0 / (i + 1)
                break
        mrr_sum += reciprocal_rank
    return mrr_sum / len(recommended_indices_list)

# ================================
# PARTE C - Geração de Recomendações
# ================================

# Carregar embeddings quantizados e converter para float32
embeddings_path = "embeddings_quantized.pt"
embeddings = torch.load(embeddings_path, map_location="cpu").float()

# Normalizar os embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)

def get_recommended_indices(query, tokenizer, model, embeddings, device="cpu", top_n=10):
    """
    Dada uma query (string), retorna os índices dos artigos mais similares
    segundo a similaridade cosseno com os embeddings pré-calculados.
    - query: string da consulta
    - tokenizer, model: PubMedBERT ou outro BERT-like carregado
    - embeddings: tensor/array com forma [num_artigos, dim_embed]
    - device: CPU ou GPU
    - top_n: quantos artigos queremos retornar

    
    """
    

    # 1) Transformar a query em embedding
    model.eval()
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Pega o vetor do [CLS] token
        query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [1, dim_embed]

    # 2) Calcular similaridade cosseno entre query_emb e todos os embeddings
    # Supondo que embeddings esteja em numpy ou tensor .cpu()
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings

    sim_scores = cosine_similarity(query_emb, embeddings_np)[0]  # array [num_artigos]

    # 3) Ordenar índices pela maior similaridade
    recommended_indices = np.argsort(sim_scores)[-top_n:][::-1]
    # agora recommended_indices contém os índices dos artigos mais similares em ordem decrescente
    return recommended_indices

# ================================
# PARTE D - Main: Fazendo o Cálculo das Métricas
# ================================

def evaluate_chatbot_metrics(test_queries, relevant_indices_list, tokenizer, model, embeddings, device="cpu", k=5, top_n=10):
    """
    Função principal que:
    1. Gera recommended_indices_list para cada query.
    2. Calcula Precision@k, MAP e MRR.
    """
    recommended_indices_list = []

    for query in test_queries:
        recommended_indices = get_recommended_indices(query, tokenizer, model, embeddings, device=device, top_n=top_n)
        recommended_indices_list.append(recommended_indices)

    # Calcula Precision@k
    precision_scores = []
    for rec_indices, rel_indices in zip(recommended_indices_list, relevant_indices_list):
        prec = precision_at_k(rec_indices, rel_indices, k)
        precision_scores.append(prec)
    mean_precision_k = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    # Calcula MAP
    map_value = mean_average_precision(recommended_indices_list, relevant_indices_list)

    # Calcula MRR
    mrr_value = mean_reciprocal_rank(recommended_indices_list, relevant_indices_list)

    return mean_precision_k, map_value, mrr_value, recommended_indices_list

# ================================
# EXEMPLO DE USO
# ================================

if __name__ == "__main__":
    # Exemplo de chamada da função:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Supondo que embeddings esteja em CPU ou GPU, verifique coerência
    # embeddings = embeddings.to("cpu")  # ou .numpy() etc.

    mean_p_k, map_score, mrr_score, recommended_indices_list = evaluate_chatbot_metrics(
        test_queries,
        relevant_indices_list,
        tokenizer,
        model,
        embeddings,
        
        device=device,
        k=10,      # P@5
        top_n=10  # Retornar top 10
    )

    print(f"Precision@5: {mean_p_k:.4f}")
    print(f"MAP: {map_score:.4f}")
    print(f"MRR: {mrr_score:.4f}")

    # Se quiser inspecionar:
    # for i, (query, rec_inds) in enumerate(zip(test_queries, recommended_indices_list)):
    #     print(f"Query: {query}\nRecommended indices: {rec_inds}\nRelevant indices: {relevant_indices_list[i]}\n")
for idx in relevant_indices_list:
    print("Índice:", idx, "Título do artigo:", artigos[idx]['Title'])
for query, rec_inds, rel_inds in zip(test_queries, recommended_indices_list, relevant_indices_list):
    print(f"\nQuery: {query}")
    print(f"Recomendados: {rec_inds}")
    print(f"Relevantes : {rel_inds}")
