import torch
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Vamos supor que você já tem tokenizer, model, embeddings, etc. carregados
# e que seu JSON está em "merged_articles_sorted.json".

test_queries = [
    "cancer treatment",             # Query 1
    "Artificial Intelligence in health",  # Query 2
    "telemedicine"        # Query 3
]

relevant_indices_list = [
    [25, 38, 118, 125, 640, 773, 793, 799, 860, 1022, 1112, 1449, 1700, 1793, 1913, 2067, 2127, 2197, 2230, 2308, 2344, 2458, 2776, 2792, 3013, 3137, 3418, 3436, 3587, 3661, 3913, 3994, 4044, 4045, 4174, 4292, 4293, 4359, 4454, 4507, 4714, 4887, 4895, 4899, 4908, 4909, 4916, 4917, 4923, 4975, 4977, 4996, 5034, 5151, 5180, 5239, 5347, 5369, 5402, 5449, 5556, 5601, 5914, 6032, 6302, 6387, 6453, 6473, 6570, 6697, 6761, 6942, 7004, 7008, 7185, 7212, 7599, 7601, 7634, 7652, 7695, 7791, 7821, 8052, 8066, 8075, 8244, 8334, 8378, 9068, 9085, 9163, 9349, 9481, 9484, 9526, 9533, 9565, 9704, 9782, 9845, 9947, 9996, 10050, 10067, 10077, 10106, 10153, 10162, 10217, 10383, 10413, 10420, 10533, 10572, 10718, 10745, 10859, 10868, 10869, 10901, 10970, 11127, 11133, 11250, 11252, 11255, 11256, 11257, 11269, 11366, 11378, 11616, 11651, 11666, 11778, 11780, 11953, 12052, 12116, 12127, 12128, 12131, 12134, 12276, 12315, 12477, 12478, 12595, 12647, 12648, 13045, 13087, 13158, 13247, 13306, 13395, 13413, 13478, 13622, 13721, 13806, 13812, 13823, 13839, 13940, 14135, 14303, 14305, 14329, 14523, 14537, 14611, 14710, 14713, 14737, 14989, 15163, 15303, 15484, 15502, 15515, 15571, 15584, 15590, 15646, 15670, 15787, 15831, 15886, 15941, 15991, 15995, 16032, 16329, 16511, 16571, 16613, 16723, 16725, 16731, 16737, 16770, 16778, 16793, 16960, 17001, 17118, 17212, 17350, 17353, 17357, 17358, 17372, 17460, 17578, 17612, 17614, 17673, 17696, 17711, 17805, 17840, 17972, 17998, 18001, 18021, 18046, 18051, 18077, 18085, 18266, 18305, 18325, 18341, 18372, 18400, 18460, 18502, 18641, 18681, 18770, 18910, 19110, 19201, 19292, 19401, 19556, 19641, 19681, 19762, 19766, 19889, 19897, 19977, 19981, 20059, 20214, 20433, 20490, 20639, 20709, 20798, 20925, 20930, 21171, 21177, 21199, 21223, 21293, 21363, 21408, 21409, 21486, 21629, 21870, 22004, 22175, 22204, 22205, 22211, 22433, 22506, 22514, 22523, 22587, 22601, 22658, 22659, 22666, 22691, 22814, 22819, 22828, 23220, 23336, 23588, 23618, 23735, 23738, 23793, 23958, 23970, 24005, 24054, 24130, 24132, 24159, 24210, 24214, 24330, 24345, 24472, 24497, 24509, 24511, 24531, 24574, 24653, 24832, 24868, 24879, 24880, 24900, 24912, 24965, 25109, 25481, 25526, 25592, 25651, 25695, 25754, 25835, 26061, 26292, 26373, 26437, 26443, 26460, 26664, 26738, 26761, 26810, 26878, 26889, 26970, 26978, 27018, 27104, 27121, 27236, 27297],
    [2098, 3144, 3145, 3146, 3251, 3314, 3315, 3316, 3317, 3318, 4191, 6538, 7968, 10224, 10774, 11400, 13676, 14073, 14090, 14964, 15137, 17542, 23090, 23194, 23366, 25100, 25959, 26927],
    [13862, 17018, 18754, 19226]
]
# Exemplo de função para achar índices relevantes
def find_articles_by_subject(input_file, subject):
    import json
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    subject_lower = subject.lower()
    relevant_indices = []
    for idx, article in enumerate(data):
        title_lower = article.get("Title","").lower()
        abstract_lower = article.get("Abstract","").lower()
        if subject_lower in title_lower or subject_lower in abstract_lower:
            relevant_indices.append(idx)
    return relevant_indices

for q in test_queries:
    # Usa a função find_articles_by_subject para gerar a lista de índices relevantes
    found_indices = find_articles_by_subject("merged_articles_sorted.json", q)
    relevant_indices_list.append(found_indices)

# Agora você tem algo como:
# relevant_indices_list[0] => todos os índices com "oncology" no título ou abstract
# relevant_indices_list[1] => todos os índices com "artificial intelligence"
# etc.

# Em seguida, precisamos gerar recommended_indices_list
# (os top artigos retornados pelo sistema).
recommended_indices_list = [
    [25, 38, 118, 125, 640, 773, 793, 799, 860, 1022, 1112, 1449, 1700, 1793, 1913, 2067, 2127, 2197, 2230, 2308, 2344, 2458, 2776, 2792, 3013, 3137, 3418, 3436, 3587, 3661, 3913, 3994, 4044, 4045, 4174, 4292, 4293, 4359, 4454, 4507, 4714, 4887, 4895, 4899, 4908, 4909, 4916, 4917, 4923, 4975, 4977, 4996, 5034, 5151, 5180, 5239, 5347, 5369, 5402, 5449, 5556, 5601, 5914, 6032, 6302, 6387, 6453, 6473, 6570, 6697, 6761, 6942, 7004, 7008, 7185, 7212, 7599, 7601, 7634, 7652, 7695, 7791, 7821, 8052, 8066, 8075, 8244, 8334, 8378, 9068, 9085, 9163, 9349, 9481, 9484, 9526, 9533, 9565, 9704, 9782, 9845, 9947, 9996, 10050, 10067, 10077, 10106, 10153, 10162, 10217, 10383, 10413, 10420, 10533, 10572, 10718, 10745, 10859, 10868, 10869, 10901, 10970, 11127, 11133, 11250, 11252, 11255, 11256, 11257, 11269, 11366, 11378, 11616, 11651, 11666, 11778, 11780, 11953, 12052, 12116, 12127, 12128, 12131, 12134, 12276, 12315, 12477, 12478, 12595, 12647, 12648, 13045, 13087, 13158, 13247, 13306, 13395, 13413, 13478, 13622, 13721, 13806, 13812, 13823, 13839, 13940, 14135, 14303, 14305, 14329, 14523, 14537, 14611, 14710, 14713, 14737, 14989, 15163, 15303, 15484, 15502, 15515, 15571, 15584, 15590, 15646, 15670, 15787, 15831, 15886, 15941, 15991, 15995, 16032, 16329, 16511, 16571, 16613, 16723, 16725, 16731, 16737, 16770, 16778, 16793, 16960, 17001, 17118, 17212, 17350, 17353, 17357, 17358, 17372, 17460, 17578, 17612, 17614, 17673, 17696, 17711, 17805, 17840, 17972, 17998, 18001, 18021, 18046, 18051, 18077, 18085, 18266, 18305, 18325, 18341, 18372, 18400, 18460, 18502, 18641, 18681, 18770, 18910, 19110, 19201, 19292, 19401, 19556, 19641, 19681, 19762, 19766, 19889, 19897, 19977, 19981, 20059, 20214, 20433, 20490, 20639, 20709, 20798, 20925, 20930, 21171, 21177, 21199, 21223, 21293, 21363, 21408, 21409, 21486, 21629, 21870, 22004, 22175, 22204, 22205, 22211, 22433, 22506, 22514, 22523, 22587, 22601, 22658, 22659, 22666, 22691, 22814, 22819, 22828, 23220, 23336, 23588, 23618, 23735, 23738, 23793, 23958, 23970, 24005, 24054, 24130, 24132, 24159, 24210, 24214, 24330, 24345, 24472, 24497, 24509, 24511, 24531, 24574, 24653, 24832, 24868, 24879, 24880, 24900, 24912, 24965, 25109, 25481, 25526, 25592, 25651, 25695, 25754, 25835, 26061, 26292, 26373, 26437, 26443, 26460, 26664, 26738, 26761, 26810, 26878, 26889, 26970, 26978, 27018, 27104, 27121, 27236, 27297],
    [2098, 3144, 3145, 3146, 3251, 3314, 3315, 3316, 3317, 3318, 4191, 6538, 7968, 10224, 10774, 11400, 13676, 14073, 14090, 14964, 15137, 17542, 23090, 23194, 23366, 25100, 25959, 26927],
    [13862, 17018, 18754, 19226]
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_recommended_indices(query, tokenizer, model, embeddings, top_n=10):
    # Gera embedding da query
    inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        query_emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
    # Similaridade com embeddings de cada artigo
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings
    sim_scores = cosine_similarity(query_emb, embeddings_np)[0]
    # Pegando top_n
    recommended_indices = sim_scores.argsort()[-top_n:][::-1]
    return recommended_indices

# Carregar PubMedBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext").to(device)
# Carregar embeddings quantizados e converter para float32
embeddings_path = "Newembeddings.pt"
embeddings = torch.load(embeddings_path, map_location="cpu").float()

# Normalizar os embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)

for q in test_queries:
    rec_indices = get_recommended_indices(q, tokenizer, model, embeddings, top_n=10)
    recommended_indices_list.append(rec_indices)

# Precisamos agora calcular Precisão@k, MAP, MRR.
# (você deve ter as funções definidas, p. ex. precision_at_k, mean_average_precision, etc.)

k = 5

def precision_at_k(recommended_indices, relevant_indices, k):
    recommended_set = set(recommended_indices[:k])
    relevant_set = set(relevant_indices)
    return len(recommended_set & relevant_set) / k if k > 0 else 0

def mean_average_precision(recommended_indices_list, relevant_indices_list):
    # mesma implementação que você tinha
    average_precisions = []
    for recommended_indices, relevant_indices in zip(recommended_indices_list, relevant_indices_list):
        score = 0.0
        num_hits = 0.0
        for i, rec_index in enumerate(recommended_indices):
            if rec_index in relevant_indices:
                num_hits += 1
                score += num_hits / (i + 1)
        average_precisions.append(score / len(relevant_indices) if relevant_indices else 0)
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0

def mean_reciprocal_rank(recommended_indices_list, relevant_indices_list):
    reciprocal_ranks = []
    for recommended_indices, relevant_indices in zip(recommended_indices_list, relevant_indices_list):
        rr = 0
        for i, rec_index in enumerate(recommended_indices):
            if rec_index in relevant_indices:
                rr = 1/(i+1)
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks)/len(reciprocal_ranks) if reciprocal_ranks else 0

# Calcular e imprimir as métricas
# Precisão@k
p_scores = []
for rec_inds, rel_inds in zip(recommended_indices_list, relevant_indices_list):
    p = precision_at_k(rec_inds, rel_inds, k)
    p_scores.append(p)
mean_p = sum(p_scores)/len(p_scores) if p_scores else 0

map_score = mean_average_precision(recommended_indices_list, relevant_indices_list)
mrr_score = mean_reciprocal_rank(recommended_indices_list, relevant_indices_list)

print(f"Precision@{k}: {mean_p:.4f}")
print(f"MAP: {map_score:.4f}")
print(f"MRR: {mrr_score:.4f}")
