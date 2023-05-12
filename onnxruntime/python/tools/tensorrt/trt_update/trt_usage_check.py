import os

# Recursively traverse all files in a directory
def list_files(dir_path):
    file_list = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".yml") or file.endswith(".yaml"):
                file_list.append(os.path.join(root, file))
    return file_list

# Search files for specific TRT keywords
def search_keywords(file_path, keywords):
    result = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if any(keyword in line.lower() for keyword in keywords):
                result.append((i + 1, line.strip()))
    return result

def main():
    dir_path = "../../../../../tools/ci_build/github/azure-pipelines/"
    output_file = "trt_usage_list.md"
    keywords = ["trt", "tensorrt"]
    file_list = list_files(dir_path)
    
    repo_url = "https://github.com/microsoft/onnxruntime/tree/main/"

    with open(output_file, "w", encoding="utf-8") as md_file:
        for file_path in file_list:
            clean_file_path = file_path.split("../")[-1].replace("\\", "/")
            search_result = search_keywords(file_path, keywords)
            if search_result:
                md_file.write(f"## {clean_file_path}\n")
                for line_num, line in search_result:
                    file_url = f"{repo_url}{clean_file_path}#L{line_num}"
                    md_file.write(f"- Line {line_num}: [`{line}`]({file_url})\n")
                md_file.write("\n")

if __name__ == "__main__":
    main()