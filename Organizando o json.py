import json

def sort_json_by_title(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)  # data deve ser uma lista de dicionários, cada um representando um artigo

    # Ordena pelo campo "Title"; se algum artigo não tiver esse campo, use .get("Title","") ou inclua um except
    data_sorted = sorted(data, key=lambda x: x["Title"] if "Title" in x else "")

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(data_sorted, out, ensure_ascii=False, indent=4)
    print(f"Arquivo '{input_file}' ordenado por título e salvo em '{output_file}'.")

# Exemplo de uso
if __name__ == "__main__":
    sort_json_by_title("merged_articles_fixed.json", "merged_articles_sorted.json")
