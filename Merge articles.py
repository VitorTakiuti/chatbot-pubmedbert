import json

def load_json_file(filename):
    """
    Load a JSON file and return its data.
    """
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)

def merge_json_files(file_list, output_file):
    """
    Merge multiple JSON files into a single JSON file, removing duplicates by title.
    """
    merged_data = []
    seen_titles = set()

    for file in file_list:
        try:
            data = load_json_file(file)
            for article in data:
                title = article.get("Title", "").strip()
                if title and title not in seen_titles:
                    merged_data.append(article)
                    seen_titles.add(title)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Save merged data to output file
    with open(output_file, "w", encoding='latin-1') as output:
        json.dump(merged_data, output, indent=4)
    print(f"Merged {len(merged_data)} articles into {output_file}")
#latin-1
def main():
    # List of JSON files to merge
    json_files = ["Oncology.json", "DigitalHealth.json", "ArtificialIntelligence.json"]
    output_file = "merged_articles.json"

    print("Merging JSON files...")
    merge_json_files(json_files, output_file)

if __name__ == "__main__":
    main()

import json

with open("merged_articles.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(json.dumps(data, ensure_ascii=False, indent=4))