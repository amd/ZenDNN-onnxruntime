import os

# Global config
dir_paths = ["../../../../../tools/ci_build/github/azure-pipelines/", "../../../../../tools/ci_build/github/linux/"]
output_file = "replacement_log.md"
repo_url = "https://github.com/microsoft/onnxruntime/tree/main/"
prompts = {
    "general": {
        "keywords_input": "Enter keywords separated by comma (press Enter to skip):\n\t",
        "replacements_input": "Enter replacement words separated by comma:\n\t"
        },
    "win_trt": {
        "old_win_trt_folder": "Enter folder path to old trt binaries. e.g TensorRT-8.6.0.12.Windows10.x86_64.cuda-11.8\n\t",
        "new_win_trt_folder": "Enter folder path to new win trt binaries which has been uploaded to Azure blob storage (press Enter to skip)\n\t e.g TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8\TensorRT-8.6.1.6, \n\t\twhich is downloadable via https://developer.nvidia.com/nvidia-tensorrt-download\n\t"
        },
    "update_docker_repository": {
        "old_trt_ver": "Enter old tensorrt version: e.g 85\n\t",
        "new_trt_ver": "Enter new tensorrt version: e.g 86\n\t"
        }
    }

# Recursively traverse all files in a directory
def list_files(dir_paths, minor_version_update=True):
    file_list = []
    for dir_path in dir_paths:
        for root, _, files in os.walk(dir_path):
            for file in files:
                for ending in [".yml", ".bat"]:
                    if file.endswith(ending):
                        file_list.append(os.path.join(root, file))
                if minor_version_update and file.startswith("Dockerfile."):
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
    # e.g trt 8.6.0 update to 8.6.1
    minor_version_update = True

    # update general keyword
    keywords_input = input(prompts["general"]["keywords_input"])
    replacements_input = input(prompts["general"]["replacements_input"])
    
    # debug only, need to be replaced with null-chk
    if not keywords_input:
        keywords_input = "8.6.0.12"
    if not replacements_input:
        replacements_input = "8.6.1.6"

    keywords = [keyword.strip() for keyword in keywords_input.split(",")]
    replacements = [replacement.strip() for replacement in replacements_input.split(",")]
    assert len(replacements) == len(keywords), "keywords and replacements should have same amount"
    
    # update win trt folder
    old_win_trt_folder = input(prompts["win_trt"]["old_win_trt_folder"])
    new_win_trt_folder = input(prompts["win_trt"]["new_win_trt_folder"])
    
    # debug only, need to be replaced with null-chk
    if not old_win_trt_folder:
        old_win_trt_folder = "TensorRT-8.6.0.12.Windows10.x86_64.cuda-11.8"
    if not new_win_trt_folder:
        new_win_trt_folder = "TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8"
    
    assert new_win_trt_folder.startswith("TensorRT-"), "Please doublecheck folder name of TRT binaries folder"
    keywords.insert(0, old_win_trt_folder)
    replacements.insert(0, new_win_trt_folder)

    # update dockerfile config
    # to-do: auto-init new dockerfile if trt new major version comes out
    # thoughts:
    # if minor update like 8.6.0->8.6.1: no new dockerfile needed, add current dockerfile to checklist and update dockerfile trt version;
    # if major update like 8.6.1->8.7.0: new dockerfile needed:
            # duplicate old dockerfile as tmp file, 
            # update old dockerfile trt version and update file name to be new
            # re-name duplicate dockerfile same as old dockerfile

    # update docker repository in yml
    old_trt_ver = input(prompts["update_docker_repository"]["old_trt_ver"])
    new_trt_ver = input(prompts["update_docker_repository"]["new_trt_ver"])
    if old_trt_ver and new_trt_ver:
        assert len(old_trt_ver) == len(new_trt_ver), "Please doublecheck\n"
        
        # null-check
        #if old_trt_ver != new_trt_ver:
        #    minor_version_update = False
        if not old_trt_ver:
            old_trt_ver = "86"
        if not new_trt_ver:
            new_trt_ver = "87"

        old_docker_repo = f"onnxruntimetensorrt{old_trt_ver}gpubuild"
        new_docker_repo = f"onnxruntimetensorrt{new_trt_ver}gpubuild"
        keywords.append(old_docker_repo)
        replacements.append(new_docker_repo)

    file_list = list_files(dir_paths, minor_version_update)
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