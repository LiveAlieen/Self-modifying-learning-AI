import os

def load_dialogue_history(file_path="dialogue_history.txt"):
    # 检查文件是否存在，如果不存在则创建
    if not os.path.exists(file_path):
        open(file_path, 'w').close()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        dialogue_history = file.read().splitlines()
    return dialogue_history

def save_dialogue_history(dialogue_history, file_path="dialogue_history.txt"):
    with open(file_path, 'w', encoding='utf-8') as file:
        for dialogue in dialogue_history:
            file.write(dialogue + '\n')