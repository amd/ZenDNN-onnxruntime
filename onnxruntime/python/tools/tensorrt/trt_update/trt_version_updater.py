import os

# Recursively traverse all files in a directory
def list_files(dir_path):
    file_list = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".yml") or file.endswith(".yaml"):
                file_list.append(os.path.join(root, file))
    return file_list

# Search files for specific TRT keywords and replace
def search_and_replace_keywords(file_path, keywords, replacements):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='utf-8') as file:
        for i, line in enumerate(lines):
            for keyword, replacement in zip(keywords, replacements):
                if keyword.lower() in line.lower():
                    line = line.replace(keyword, replacement)
                    result.append((i + 1, line.strip()))
            file.write(line)
    return result

def exec(output_file):
    dir_path = "../../../../../tools/ci_build/github/azure-pipelines/"
    keywords_input = input("Please enter keywords separated by comma: ")
    replacements_input = input("Please enter replacement words separated by comma: ")
    keywords = [keyword.strip() for keyword in keywords_input.split(",")]
    replacements = [replacement.strip() for replacement in replacements_input.split(",")]
    file_list = list_files(dir_path)
    
    repo_url = "https://github.com/microsoft/onnxruntime/tree/main/"

    with open(output_file, "w", encoding="utf-8") as md_file:
        for file_path in file_list:
            replaced_result = search_and_replace_keywords(file_path, keywords, replacements)
            if replaced_result:
                md_file.write(f"## {os.path.relpath(file_path, dir_path)}\n")
                for line_num, line in replaced_result:
                    relative_file_path = os.path.relpath(file_path, dir_path)
                    file_url = f"{repo_url}{relative_file_path}#L{line_num}"
                    md_file.write(f"- Line {line_num} (replaced): [`{line}`]({file_url})\n")
                md_file.write("\n")

def main():
    output_file = "log.md"
    exec(output_file)

if __name__ == "__main__":
    main()