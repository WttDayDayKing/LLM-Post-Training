import contextlib
import numpy as np
import torch
from tqdm import tqdm
import json
from typing import  Sequence
from collections.abc import Iterable
import os
from torch.utils.data import Dataset
import random

class LoadDataset(Dataset):
    def __init__(self, all_file_paths, tokenizer, max_seq_length=1024, sample_percentage=1.0, seed=0):
        self.file_paths_list = load_train_files(all_file_paths)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sample_percentage = sample_percentage
        self.data_indices = self._load_data_indices()
        self.data_indices = random.sample(self.data_indices, int(len(self.data_indices) * self.sample_percentage))
        self.prompt = ''
        self.answer_prefix = '\nAnswer: '
        self.pmt_len = 128
        self.ans_len = 128
    
    def _load_data_indices(self):
        data_indices = []
        for file_path in self.file_paths_list:
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    data_indices.append((file_path, offset))
                    
        return data_indices

    def __len__(self):
        return len(self.data_indices)
    
    # def __getitem__(self, idx):
    #     file_path, line_idx = self.data_indices[idx]
    #     with open(file_path, 'r') as f:
    #         for current_idx, line in enumerate(f):
    #             if current_idx == line_idx:
    #                 return self.tokenizer(line)

    ####单轮对话
    # def __getitem__(self, idx):
    #     file_path, offset = self.data_indices[idx]
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         f.seek(offset)
    #         line = f.readline()
    #         sample = json.loads(line.strip())
    #         #原始代码IDEAL
    #         question = sample['instruction'] + ' ' + sample['input']
    #         answer = '\n### ' + sample['output']
    #         ###适配我们的数据集格式
    #         # 从 messages 中提取用户提问和助手回答
    #         messages = sample['messages']
    #         user_msg = next(msg for msg in messages if msg['role'] == 'user')
    #         assistant_msg = next(msg for msg in messages if msg['role'] == 'assistant')
            
    #         question = user_msg['content']
    #         answer = '\n### ' + assistant_msg['content']   # 保留你原有的回答前缀格式
    
            
    #         input_text = self.prompt + question + self.answer_prefix
    #         target_text = answer + self.tokenizer.eos_token

    #         if self.tokenizer.pad_token is None:
    #             self.tokenizer.pad_token = self.tokenizer.eos_token
    #             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    #             # print("Pad token ID:", self.tokenizer.pad_token_id)

    #         encoding = self.tokenizer(
    #             input_text + target_text,
    #             add_special_tokens=False,
    #             padding='max_length',
    #             truncation=True,
    #             max_length=self.pmt_len + self.ans_len,
    #             return_tensors='pt'
    #         )

    #         labels = encoding['input_ids'].squeeze().clone()
    #         labels[labels == self.tokenizer.pad_token_id] = -100
    #         labels[:self.pmt_len] = -100

    #         return {
    #             'input_ids': encoding['input_ids'].squeeze(),
    #             'attention_mask': encoding['attention_mask'].squeeze(),
    #             'labels': labels
    #         }

    ####适配多轮对话
    def __getitem__(self, idx):
        file_path, offset = self.data_indices[idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            sample = json.loads(line.strip())
            
            messages = sample['messages'] # 此时包含多轮对话的列表

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # 1. 动态构建多轮文本与 Labels 掩码
            full_input_ids = []
            full_labels = []
            

            for msg in messages:
                role = msg['role']
                content = msg['content']
                
                if role == 'user':
                    # 拼接 User 文本，多轮中建议加上明确的角色标志（这里以自定义前缀为例，也可以用 chat_template）
                    text = self.prompt + content + self.answer_prefix
                    text_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    
                    full_input_ids.extend(text_ids)
                    # User 的输入不计算 Loss，全部用 -100 掩码
                    full_labels.extend([-100] * len(text_ids))
                    
                elif role == 'assistant':
                    # 拼接 Assistant 文本，并带上结束符
                    text = '\n### ' + content + self.tokenizer.eos_token
                    text_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    
                    full_input_ids.extend(text_ids)
                    # Assistant 的回复需要计算 Loss，Label 保持与 Input ID 一致
                    full_labels.extend(text_ids)

            # 2. 截断与填充 (Truncation & Padding)
            max_total_len = self.pmt_len + self.ans_len  # 你的目标最大长度
            
            # 截断
            if len(full_input_ids) > max_total_len:
                full_input_ids = full_input_ids[:max_total_len]
                full_labels = full_labels[:max_total_len]
                attention_mask = [1] * max_total_len
            else:
                # 填充
                padding_len = max_total_len - len(full_input_ids)
                attention_mask = [1] * len(full_input_ids) + [0] * padding_len
                
                full_input_ids = full_input_ids + [self.tokenizer.pad_token_id] * padding_len
                # 填充位不计算 Loss，设为 -100
                full_labels = full_labels + [-100] * padding_len

            import torch
            return {
                'input_ids': torch.tensor(full_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(full_labels, dtype=torch.long)
            }
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def load_train_files(all_file_paths):
    file_path_list = []
    exclude_dirs=['Multi-lang-Knowledge','Multi-lang-Math']
    has_subfolders = any(os.path.isdir(os.path.join(all_file_paths, item)) for item in os.listdir(all_file_paths))
    if has_subfolders:
        for dirpath, dirnames, filenames in os.walk(all_file_paths):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for filename in filenames:
                if filename.endswith('.jsonl'):
                    filepath = os.path.join(dirpath, filename)
                    file_path_list.append(filepath)
    else:
        for filename in os.listdir(all_file_paths):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(all_file_paths, filename)
                file_path_list.append(filepath)
    return file_path_list