
import json

def find_articles_by_subject(input_file, subject):
    """
    Lê o arquivo JSON e retorna os índices dos artigos que contêm
    o 'subject' no título ou no abstract (ignora maiúsculas e minúsculas).
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # data deve ser uma lista de dicionários
    
    subject_lower = subject.lower()
    relevant_indices = []

    for idx, article in enumerate(data):
        title = article.get("Title", "")
        abstract = article.get("Abstract", "")
        
        # Normaliza para lowercase
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        
        # Se o "subject" estiver presente no título ou no abstract, marcamos o índice
        if subject_lower in title_lower or subject_lower in abstract_lower:
            relevant_indices.append(idx)
    
    return relevant_indices

def main():
    # Nome do arquivo JSON contendo os artigos
    json_file = "merged_articles_fixed.json"  
    # Exemplo de termo ou assunto que queremos pesquisar
    subject = "telemedicine applications"  
    
    found_indices = find_articles_by_subject(json_file, subject)
    
    print(f"Número de artigos encontrados com '{subject}': {len(found_indices)}")
    print("Índices relevantes:", found_indices)

    # Se quiser imprimir título dos artigos, por exemplo:
    
    #with open(json_file, "r", encoding="utf-8") as f:
        #data = json.load(f)
        #for idx in found_indices:
            #print(f"Index: {idx} | Title: {data[idx].get('Title','No Title')}")
  

if __name__ == "__main__":
    main()


