import json

with open("merged_articles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

#print(json.dumps(data, ensure_ascii=False, indent=4))

# Agora salva em outro arquivo, sem caracteres escapados:
with open("merged_articles_fixed.json", "w", encoding="utf-8") as out_file:
    json.dump(data, out_file, ensure_ascii=False, indent=4)