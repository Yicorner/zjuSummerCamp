import json
    
    
def save_data_to_disk(dir_path, data):
    with open(dir_path + 'data_score_correct.json', 'w') as f:
        json.dump(data, f)
        print("save data_score_correct success")
        
def load_data_to_memory(dir_path):
    with open(dir_path + 'data_score_correct.json', 'r') as f:
        loaded_data = json.load(f)
        print("load data_score_correct success")
    return loaded_data