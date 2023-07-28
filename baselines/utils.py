import os
import torch
import random
import json
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
def collate_fn(data):

    support = data[0][0]
    query = data[0][1]
    id2label = data[0][2]
    # batch_support = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'text-mask': [], 'att-mask': []}
    # batch_query = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'text-mask': [], 'att-mask': []}
    # batch_support = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'att-mask': []}
    # batch_query = {'tokens': [], 'trigger_label': [], 'B-mask':[], 'I-mask': [], 'att-mask': []}  
    # batch_id2label = []
    
    # support_sets, query_sets, id2labels = zip(*data)
    
    # for i in range(len(support_sets)):
    #     for k in support_sets[i]:
    #         batch_support[k].append(support_sets[i][k])
    #     for k in query_sets[i]:
    #         batch_query[k].append(query_sets[i][k])
    #     batch_id2label.append(id2labels[i])
    
    # for k in batch_support:
    #     batch_support[k] = torch.cat(batch_support[k], 0)
    # for k in batch_query:
    #     batch_query[k] = torch.cat(batch_query[k], 0)
    
    return support, query, id2label

class FewEventDataset(Dataset):
    def __init__(self, 
                 raw_data, 
                 tokenizer,
                 max_length = 128, 
                 N=5, K=5, Q=1):
        self.raw_data = raw_data
        self.classes = self.raw_data.keys()
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        self.N = N
        self.K = K
        self.Q = Q

    def __build_dict__(self, event_type_list):
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
    
    def __sample__(self, target_classes, sample_num):
        count = { c:0 for c in target_classes }
        sample_set = []

        for event_type in target_classes:
            samples = random.sample(self.raw_data[event_type], sample_num)
            sample_set.extend(samples)
            for sample in samples:
                for e in sample["events"]:
                    if e["event_type"] in count.keys():
                        count[e["event_type"]] += 1
        for s in sample_set:
            sample_set.remove(s)
            tmp_count = count.copy()
            for e in s["events"]:
                if e["event_type"] in tmp_count.keys():
                    tmp_count[e["event_type"]] -= 1
            is_reserve = sorted(tmp_count.values())[0] < sample_num
            if is_reserve:
                sample_set.append(s)
            else:
                count = tmp_count
        return sample_set
    
    def preprocess(self, instance, target_classes):
        
        tokens = instance["tokens"]
        trigger_label = ['O'] * len(tokens)
        # B_mask = [0] * len(tokens)
        # I_mask = [0] * len(tokens)
        # text_mask = [1] * len(tokens)

        for event in instance["events"]:
            event_type = event["event_type"]
            if event_type not in target_classes:
                continue
            trigger_start_pos = event["start"]
            trigger_end_pos = event["end"]
            for i in range(trigger_start_pos, trigger_end_pos):
                if i == trigger_start_pos:
                    trigger_label[i] = f"B-{event_type}"
                    #B_mask[i] = 1
                else:
                    trigger_label[i] = f"I-{event_type}"
                    #I_mask[i] = 1
        # result = {'tokens': tokens, 'trigger_label': trigger_label, 'B-mask': B_mask, 'I-mask': I_mask, 'text-mask': text_mask}
        result = {'tokens': tokens, 'trigger_label': trigger_label}
        return result
    
    def tokenize(self, instance, label2id):

        raw_tokens = instance['tokens']
        raw_label = instance['trigger_label']
        # raw_B_mask = instance['B-mask']
        # raw_I_mask = instance['I-mask']
        # raw_text_mask = instance['text-mask']

        # token -> index
        # tokens = ['[CLS]']
        # label = ['O']
        # B_mask = [0]
        # I_mask = [0]
        # text_mask = [0]
        tokens, labels = [], []
        for i, token in enumerate(raw_tokens):
            tokenize_result = self.tokenizer.tokenize(token)
            tokens += tokenize_result

            if len(tokenize_result) > 1:
                labels += [raw_label[i]]
                # B_mask += [raw_B_mask[i]]
                # I_mask += [raw_I_mask[i]]
                # text_mask += [raw_text_mask[i]]

                if raw_label[i][0] == "B":
                    tmp_label = "I" + raw_label[i][1:]
                    labels += [tmp_label] * (len(tokenize_result) - 1)
                    # B_mask += [0] * (len(tokenize_result) - 1)
                    # I_mask += [1] * (len(tokenize_result) - 1)
                    # text_mask += [0] * (len(tokenize_result) - 1)

                else:
                    labels += [raw_label[i]] * (len(tokenize_result) - 1)
                    # B_mask += [raw_B_mask[i]] * (len(tokenize_result) - 1)
                    # I_mask += [raw_I_mask[i]] * (len(tokenize_result) - 1)
                    # text_mask += [raw_text_mask[i]] * (len(tokenize_result) - 1)
            else:
                labels += [raw_label[i]] * len(tokenize_result)
                # B_mask += [raw_B_mask[i]] * len(tokenize_result)
                # I_mask += [raw_I_mask[i]] * len(tokenize_result)
                # text_mask += [raw_text_mask[i]] * len(tokenize_result)
        assert len(tokens) == len(labels)

        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
            labels = labels[:self.max_length - 2]
        tokens = ['[CLS]'] + tokens
        labels = ['PAD'] + labels
        tokens += ['[SEP]']
        labels.append('PAD')
        # B_mask.append(0)
        # I_mask.append(0)
        # text_mask.append(0)

        # max_length = self.max_length
        # att mask
        attention_mask = torch.zeros(self.max_length)
        attention_mask[:len(tokens)] = 1

        token_type_ids = torch.zeros(self.max_length)

        # padding
        while len(tokens) < self.max_length:
            tokens.append('[PAD]')
            labels.append('PAD')
            # B_mask.append(0)
            # I_mask.append(0)
            # text_mask.append(0)

        # tokens = tokens[:max_length]
        # label = label[:max_length]
        # B_mask = B_mask[:max_length]
        # I_mask = I_mask[:max_length]
        # text_mask = text_mask[:max_length]
        
        # to ids
        input_ids  = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids  = torch.tensor(input_ids).long()

        label_ids = list(map(lambda x: label2id[x], labels))
        label_ids = torch.tensor(label_ids).long()

        attention_mask = attention_mask.long()
        token_type_ids = token_type_ids.long()
        # B_mask_ids = torch.tensor(B_mask)
        # I_mask_ids = torch.tensor(I_mask)
        # text_mask_ids = torch.tensor(text_mask)
        return input_ids, attention_mask, token_type_ids, label_ids

        # return token_ids, label_ids, B_mask_ids, I_mask_ids, text_mask_ids, att_mask


    def __len__(self):
        return 99999999

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        label2id, id2label = self.__build_dict__(target_classes)
        
        # support_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], 'text-mask': [], "att-mask": []}
        # query_set = {'tokens': [], 'trigger_label': [], 'B-mask': [], 'I-mask': [], 'text-mask': [], "att-mask": []}

        support_set = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'label_ids': [] }
        query_set = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'label_ids': [] }

        support_samples = self.__sample__(target_classes, self.K)
        for sample in support_samples:
            instance = self.preprocess(sample, target_classes)
            # token_ids, label_ids, B_mask, I_mask, text_mask, att_mask = self.tokenize(instance, label2id)
            input_ids, attention_mask, token_type_ids, label_ids = self.tokenize(instance, label2id)
            support_set['input_ids'].append(input_ids)
            support_set['attention_mask'].append(attention_mask)
            support_set['token_type_ids'].append(token_type_ids)
            support_set['label_ids'].append(label_ids)

            # support_set['tokens'].append(token_ids)
            # support_set['trigger_label'].append(label_ids)
            # support_set['B-mask'].append(B_mask)
            # support_set['I-mask'].append(I_mask)
            # support_set['text-mask'].append(text_mask)
            # support_set['att-mask'].append(att_mask)
        
        query_samples = self.__sample__(target_classes, self.Q)
        for sample in query_samples:
            instance = self.preprocess(sample, target_classes)
            # token_ids, label_ids, B_mask, I_mask, text_mask, att_mask = self.tokenize(instance, label2id)
            input_ids, attention_mask, token_type_ids, label_ids = self.tokenize(instance, label2id)
            query_set['input_ids'].append(input_ids)
            query_set['attention_mask'].append(attention_mask)
            query_set['token_type_ids'].append(token_type_ids)
            query_set['label_ids'].append(label_ids)         
            # query_set['tokens'].append(token_ids)
            # query_set['trigger_label'].append(label_ids)
            # query_set['B-mask'].append(B_mask)
            # query_set['I-mask'].append(I_mask)
            # query_set['text-mask'].append(text_mask)
            # query_set['att-mask'].append(att_mask)   

        for k, v in support_set.items():
            support_set[k] = torch.stack(v)
        
        for k, v in query_set.items():
            query_set[k] = torch.stack(v)
        
        return support_set, query_set, id2label


def load_dataset(args):
    dataset = {}
    modes = ["train","dev","test"]
    for mode in modes:
        data_path = os.path.join(args.data_dir,mode+".json")
        raw_data = json.load(open(data_path,"r"))
        dataset[mode] = FewEventDataset(raw_data,
                                        args.tokenizer, 
                                        args.max_len, 
                                        args.N, args.K, args.Q)
    return dataset

if __name__ == '__main__':
    raw_data = json.load(open("./data/maven/dev.json","r"))

    tokenizer = AutoTokenizer.from_pretrained("../bert/bert-base-uncased")
    dataset = FewEventDataset(raw_data,tokenizer)
    dataloader = DataLoader(dataset=dataset["train"],
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=collate_fn)

    dataloader = iter(dataloader)

    support_set, query_set, id2label = next(dataloader)