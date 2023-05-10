import os
import json
import random
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset

def collate_fn(data):
    batch_support = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'text-mask': [], 'att-mask': []}
    batch_query = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'text-mask': [], 'att-mask': []}
    # batch_support = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'att-mask': []}
    # batch_query = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'att-mask': []}  
    batch_id2label = []
    
    support_sets, query_sets, id2labels = zip(*data)
    
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k].append(support_sets[i][k])
        for k in query_sets[i]:
            batch_query[k].append(query_sets[i][k])
        batch_id2label.append(id2labels[i])
    
    for k in batch_support:
        batch_support[k] = torch.cat(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.cat(batch_query[k], 0)
    
    return batch_support, batch_query, batch_id2label

# def get_loader(args, mode):
    
#     data_dir = args.data_dir
#     max_len = args.max_len
#     tokenizer = args.tokenizer
#     K, Q = args.K, args.Q
#     batch_size = args.batch_size
    
#     if mode == "train":
#         N = args.trainN
#         data_file = "meta_train_dataset.json"
#     elif mode == "dev":
#         N = args.evalN
#         data_file = "meta_dev_dataset.json"
#     elif mode == "test":
#         N = args.evalN
#         data_file = "meta_test_dataset.json"
#     else:
#         raise ValueError("Error mode!")
    
    
#     dataset_path = os.path.join(data_dir, data_file)
#     dataset = FewEventDataset(dataset_path, 
#                                   max_len, 
#                                   tokenizer,
#                                   N, K, Q)
        
#     dataloader = DataLoader(dataset=dataset,
#                             batch_size=batch_size,
#                             shuffle=False,
#                             pin_memory=True,
#                             collate_fn=collate_fn)
#     return iter(dataloader)

def load_dataset(args):
    dataset = {}
    if args.split:
        train_data, dev_data, test_data = split_json_data(args)
        dataset['train'] = FewEventDataset(train_data, 
                                      args.max_len, 
                                      args.tokenizer,
                                      args.trainN, args.K, args.Q)

        dataset['dev'] = FewEventDataset(dev_data, 
                                      args.max_len, 
                                      args.tokenizer,
                                      args.evalN, args.K, args.Q)

        dataset['test'] = FewEventDataset(test_data, 
                                      args.max_len, 
                                      args.tokenizer,
                                      args.evalN, args.K, args.Q)
    else:
        modes = ["train","dev","test"]
        for mode in modes:
            dataset_path = os.path.join(args.data_dir, 'meta_{}_dataset.json'.format(mode))
            if mode == "train":
                N = args.trainN
            else:
                N = args.evalN
            raw_data = json.load(open(dataset_path, "r"))
            dataset[mode] = FewEventDataset(raw_data, 
                                      args.max_len, 
                                      args.tokenizer,
                                      N, args.K, args.Q)
    return dataset

def split_json_data(args):
    data_dir = args.data_dir
    dataset = data_dir.split('/')[-1]
    dataset_path = os.path.join(data_dir, 'all_data.json')
    json_data = json.load(open(dataset_path, "r"))
    train_data, dev_data, test_data = {}, {}, {}
    event_types = list(json_data.keys())
    random.shuffle(event_types)
    if dataset == 'fewevent':
        train_types = event_types[:80]
        dev_types = event_types[80: 90]
        test_types = event_types[90: 100]
    elif dataset == 'ace':
        train_types = event_types[:13]
        dev_types = event_types[13: 23]
        test_types = event_types[23: 33]
    elif dataset == 'maven':
        train_types = event_types[:64]
        dev_types = event_types[64: 80]
        test_types = event_types[80: 100]
    else:
        raise NotImplementedError
    
    for k in train_types:
        train_data[k] = json_data[k]
    for k in dev_types:
        dev_data[k] = json_data[k]
    for k in test_types:
        test_data[k] = json_data[k]
    return train_data, dev_data, test_data

class FewEventDataset(Dataset):
    def __init__(self, 
                 raw_data, 
                 max_length, 
                 tokenizer,
                 N, K, Q):
        self.raw_data = raw_data
        # self.raw_data = json.load(open(dataset_path, "r"))
        self.classes = self.raw_data.keys()
        
        # self.max_length = max_length - 2
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.N = N
        self.K = K
        self.Q = Q
        
    def __len__(self):
        return 99999999
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        label2id, id2label = self.build_dict(target_classes)
        
        support_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], 'text-mask': [], "att-mask": []}
        query_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], 'text-mask': [], "att-mask": []}
        # support_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], "att-mask": []}
        # query_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], "att-mask": []}       
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.raw_data[class_name]))), 
                    self.K + self.Q, False)
            
            count = 0
            for j in indices:
                if count < self.K:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name)
                    token_ids, label_ids, B_mask, I_mask, text_mask, att_mask = self.tokenize(instance, label2id)
                    
                    support_set['tokens'].append(token_ids)
                    support_set['trigger_label'].append(label_ids)
                    support_set['B-mask'].append(B_mask)
                    support_set['I-mask'].append(I_mask)
                    support_set['text-mask'].append(text_mask)
                    support_set['att-mask'].append(att_mask)
                else:
                    instance = self.preprocess(self.raw_data[class_name][j], class_name)
                    token_ids, label_ids, B_mask, I_mask, text_mask, att_mask = self.tokenize(instance, label2id)
                    
                    query_set['tokens'].append(token_ids)
                    query_set['trigger_label'].append(label_ids)
                    query_set['B-mask'].append(B_mask)
                    query_set['I-mask'].append(I_mask)
                    query_set['text-mask'].append(text_mask)
                    query_set['att-mask'].append(att_mask)
                count += 1
        
        for k, v in support_set.items():
            support_set[k] = torch.stack(v)
        
        for k, v in query_set.items():
            query_set[k] = torch.stack(v)
        
        return support_set, query_set, id2label


    def preprocess(self, instance, event_type):
        result = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'text-mask': []}
        # result = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': []}

        sentence = instance['tokens']
        result['tokens'] = sentence
        
        trigger_label = ['O'] * len(sentence)
        B_mask = [0] * len(sentence)
        I_mask = [0] * len(sentence)
        text_mask = [1] * len(sentence)

        trigger_length = len(instance['trigger'])

        trigger_start_pos = instance['position'][0]
        trigger_end_pos = trigger_start_pos + trigger_length
        for i in range(trigger_start_pos, trigger_end_pos):
            if i == trigger_start_pos:
                trigger_label[i] = f"B-{event_type}"
                B_mask[i] = 1
            else:
                trigger_label[i] = f"I-{event_type}"
                I_mask[i] = 1

        result['trigger_label'] = trigger_label
        result['B-mask'] = B_mask
        result['I-mask'] = I_mask
        result['text-mask'] = text_mask
        
        return result
    
    def build_dict(self, event_type_list):
        label2id = OrderedDict()
        id2label = OrderedDict()
        
        label2id['O'] = 0
        id2label[0] = 'O'
        label2id['PAD'] = -100
        id2label[-100] = 'PAD'
        for i, event_type in enumerate(event_type_list):
            label2id['B-' + event_type] = 2*i + 1
            label2id['I-' + event_type] = 2*i + 2
            id2label[2*i + 1] = 'B-' + event_type
            id2label[2*i + 2] = 'I-' + event_type
        
        return label2id, id2label
    
    def tokenize(self, instance, label2id):
        max_length = self.max_length

        raw_tokens = instance['tokens']
        raw_label = instance['trigger_label']
        raw_B_mask = instance['B-mask']
        raw_I_mask = instance['I-mask']
        raw_text_mask = instance['text-mask']

        # token -> index
        tokens = ['[CLS]']
        label = ['O']
        B_mask = [0]
        I_mask = [0]
        text_mask = [0]
        for i, token in enumerate(raw_tokens):
            tokenize_result = self.tokenizer.tokenize(token)
            tokens += tokenize_result

            if len(tokenize_result) > 1:
                label += [raw_label[i]]
                B_mask += [raw_B_mask[i]]
                I_mask += [raw_I_mask[i]]
                text_mask += [raw_text_mask[i]]

                if raw_label[i][0] == "B":
                    tmp_label = "I" + raw_label[i][1:]
                    label += [tmp_label] * (len(tokenize_result) - 1)
                    B_mask += [0] * (len(tokenize_result) - 1)
                    I_mask += [1] * (len(tokenize_result) - 1)
                    text_mask += [0] * (len(tokenize_result) - 1)

                else:
                    label += [raw_label[i]] * (len(tokenize_result) - 1)
                    B_mask += [raw_B_mask[i]] * (len(tokenize_result) - 1)
                    I_mask += [raw_I_mask[i]] * (len(tokenize_result) - 1)
                    text_mask += [raw_text_mask[i]] * (len(tokenize_result) - 1)
            else:
                label += [raw_label[i]] * len(tokenize_result)
                B_mask += [raw_B_mask[i]] * len(tokenize_result)
                I_mask += [raw_I_mask[i]] * len(tokenize_result)
                text_mask += [raw_text_mask[i]] * len(tokenize_result)

        tokens += ['[SEP]']
        label.append('O')
        B_mask.append(0)
        I_mask.append(0)
        text_mask.append(0)

        # att mask
        att_mask = torch.zeros(max_length)
        att_mask[:len(tokens)] = 1


        # padding
        while len(tokens) < self.max_length:
            tokens.append('[PAD]')
            label.append('PAD')
            B_mask.append(0)
            I_mask.append(0)
            text_mask.append(0)
        
        tokens = tokens[:max_length]
        label = label[:max_length]
        B_mask = B_mask[:max_length]
        I_mask = I_mask[:max_length]
        text_mask = text_mask[:max_length]

        # tokens[-1] = '[SEP]'

        # to ids
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_ids = torch.tensor(token_ids).long()

        label_ids = list(map(lambda x: label2id[x], label))
        label_ids = torch.tensor(label_ids).long()

        B_mask_ids = torch.tensor(B_mask)
        I_mask_ids = torch.tensor(I_mask)
        text_mask_ids = torch.tensor(text_mask)

        return token_ids, label_ids, B_mask_ids, I_mask_ids, text_mask_ids, att_mask

