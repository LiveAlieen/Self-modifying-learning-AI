from dialogue_history import load_dialogue_history, save_dialogue_history
from github_integration import join_github_project
from neural_network import build_model, online_learning

def process_dialogue(user_input):
    # 示例回复，实际应根据对话内容生成
    return "处理对话后的回复"

def self_modify(dialogue_history, user_input):
    dialogue_history.append(user_input)
    return dialogue_history

if __name__ == "__main__":
    dialogue_history_file = "dialogue_history.txt"
    dialogue_history = load_dialogue_history(dialogue_history_file)
    
    while True:
        user_input = input("你的请求：")
        if user_input.lower() == 'exit':  # 添加退出机制
            print("退出程序。")
            break
        response = process_dialogue(user_input)
        print("AI的回答：", response)
        
        dialogue_history = self_modify(dialogue_history, user_input)
        save_dialogue_history(dialogue_history)  # 修正：移除多余的参数
        
        # 假设online_learning需要dialogue_history作为参数
        online_learning(dialogue_history)