import os

# Global config
dir_path = "../../../../../tools/ci_build/github/azure-pipelines/"
output_file = "replacement_log.md"
repo_url = "https://github.com/microsoft/onnxruntime/tree/main/"
prompts = {
    "general": {
        "keywords_input": "Enter keywords separated by comma (press Enter to skip):\n\t",
        "replacements_input": "Enter replacement words separated by comma:\n\t"
        },
    "win_trt": {
        "old_win_trt_folder": "Enter folder path to old trt binaries. e.g TensorRT-8.5.3.1.Windows10.x86_64.cuda-11.8.cudnn8.6\n\t",
        "new_win_trt_folder": "Enter folder path to new win trt binaries which has been uploaded to Azure blob storage (press Enter to skip)\n\t e.g TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6, \n\t\twhich is downloadable via https://developer.nvidia.com/nvidia-tensorrt-download\n\t"
        }
    }

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

def updater():
    # update general keyword
    keywords_input = input(prompts["general"]["keywords_input"])
    replacements_input = input(prompts["general"]["replacements_input"])
    keywords = [keyword.strip() for keyword in keywords_input.split(",")]
    replacements = [replacement.strip() for replacement in replacements_input.split(",")]
    assert len(replacements) == len(keywords), "keywords and replacements should have same amount"
    
    # update win trt folder
    old_win_trt_folder = input(prompts["win_trt"]["old_win_trt_folder"])
    new_win_trt_folder = input(prompts["win_trt"]["new_win_trt_folder"])
    assert new_win_trt_folder.startswith("TensorRT-"), "Please doublecheck folder name of TRT binaries folder"
    keywords.insert(0, old_win_trt_folder)
    replacements.insert(0, new_win_trt_folder)

    file_list = list_files(dir_path)
    with open(output_file, "w", encoding="utf-8") as md_file:
        for file_path in file_list:
            clean_file_path = file_path.split("../")[-1].replace("\\", "/")
            replaced_result = search_and_replace_keywords(file_path, keywords, replacements)
            if replaced_result:
                md_file.write(f"## {clean_file_path}\n")
                for line_num, line in replaced_result:
                    file_url = f"{repo_url}{clean_file_path}#L{line_num}"
                    md_file.write(f"- Line {line_num} (replaced): [`{line}`]({file_url})\n")
                md_file.write("\n")

def main():
    updater()

if __name__ == "__main__":
    main()