import json
    
    
def save_data_to_disk(dir_path, data):
    with open(dir_path + 'data_score_correct.json', 'w') as f:
        json.dump(data, f)
        print(f"you have saved data and figure in {dir_path} successfully")
        
def load_data_to_memory(dir_path):
    with open(dir_path + 'data_score_correct.json', 'r') as f:
        loaded_data = json.load(f)
    return loaded_data