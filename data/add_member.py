import os, json, random
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

SHADOW_NUM = 10

def load_data(dir):
    with open(dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data  

def save_data(dir, obj):
    with open(dir, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def main():
    shadow_data_dir = "./shadow"
    raw_data_dir = "./wiki_json"
    os.makedirs(shadow_data_dir, exist_ok=True)
    shadow_data_raw = [] 
    for i in range(SHADOW_NUM):
        shadow_data_raw.append(load_data(os.path.join(raw_data_dir, "train_finetune_{}.json".format(i))))
    train_member = load_data("./data/train/test.json") 
    val_member = load_data("./data/val/test.json") 
    final_member = load_data("./data/final/test.json") 
    train_record = [[] for i in range(SHADOW_NUM)]
    for entry in train_member:
        indice = random.sample(range(SHADOW_NUM), SHADOW_NUM//2) 
        for i in range(SHADOW_NUM):
            if i in indice:
                shadow_data_raw[i].append({"text":entry}) 
                train_record[i].append(True) 
            else:
                train_record[i].append(False) 
    save_data(os.path.join(shadow_data_dir, "train_member.json"), train_record)
    val_record = [[] for i in range(SHADOW_NUM)]
    for entry in val_member:
        indice = random.sample(range(SHADOW_NUM), SHADOW_NUM//2) 
        for i in range(SHADOW_NUM):
            if i in indice:
                shadow_data_raw[i].append({"text":entry}) 
                val_record[i].append(True) 
            else:
                val_record[i].append(False) 
    save_data(os.path.join(shadow_data_dir, "val_member.json"), val_record)
    final_record = [[] for i in range(SHADOW_NUM)]
    for entry in final_member:
        indice = random.sample(range(SHADOW_NUM), SHADOW_NUM//2) 
        for i in range(SHADOW_NUM):
            if i in indice:
                shadow_data_raw[i].append({"text":entry}) 
                final_record[i].append(True) 
            else:
                final_record[i].append(False) 
    save_data(os.path.join(shadow_data_dir, "final_member.json"), final_record)
    for i in range(SHADOW_NUM):
        save_data(os.path.join(shadow_data_dir, "shadow_{}.json".format(i)), shadow_data_raw[i])
    
    
    

if __name__ == "__main__":
    main()