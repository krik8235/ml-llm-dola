import json
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, max_len: int = 256, data_list: list[dict] = list()):
        self.max_len = max_len

        if not data_list:
            file_path = 'data/sample_questions.jsonl'
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data_list.append(json.loads(line.strip()))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int = 0) -> tuple[str, str]:
        item = self.data_list[idx]
        question, label = item['question'], item['label'] 
        return question, label
    
    def get_all_questions(self) -> list[str]:
        items = self.data_list
        return [item['question'] for item in items]
    

    def get_all_labels_and_categories(self) -> tuple[list[str], list[int]]:
        items = self.data_list
        return [item['label'] for item in items], [int(item['category']) for item in items], 
    