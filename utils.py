import os
import torch
import random
import json
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import wordnet as wn
from nltk import pos_tag

# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None
    
def collate_fn(data):

    support = data[0][0]
    query = data[0][1]
    id2label = data[0][2]

    
    return support, query, id2label

class FewEventDataset(Dataset):
    def __init__(self, 
                 raw_data, 
                 tokenizer,
                 framenet,
                 max_length = 128, 
                 N=5, K=5, Q=1):
        self.raw_data = raw_data
        self.classes = self.raw_data.keys()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.framenet = framenet
        
        self.N = N
        self.K = K
        self.Q = Q

    def __build_dict__(self, target_classes):
        label2id = OrderedDict()
        id2label = OrderedDict()
        
        label2id = { j:i+1 for i,j in enumerate(target_classes)}
        id2label = { i+1:j for i,j in enumerate(target_classes)}
        label2id['O'] = 0
        id2label[0] = 'O'
        label2id['PAD'] = -100
        id2label[-100] = 'PAD'
        
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

        for event in instance["events"]:
            event_type = event["event_type"]
            if event_type not in target_classes:
                continue
            trigger_start_pos = event["start"]
            trigger_end_pos = event["end"]

            for i in range(trigger_start_pos, trigger_end_pos):
                trigger_label[i] = event_type

        result = {'tokens': tokens, 'trigger_label': trigger_label}
        return result
    
    def tokenize(self, instance, label2id):

        raw_tokens = instance['tokens']
        raw_label = instance['trigger_label']

        tokens, labels = [], []
        for i, token in enumerate(raw_tokens):
            tokenize_result = self.tokenizer.tokenize(token)
            tokens += tokenize_result
            labels += [raw_label[i]] * len(tokenize_result)

        assert len(tokens) == len(labels)

        if len(tokens) > self.max_length - 2:
            tokens = tokens[:self.max_length - 2]
            labels = labels[:self.max_length - 2]
        tokens = ['[CLS]'] + tokens
        labels = ['O'] + labels
        tokens += ['[SEP]']
        labels.append('O')

        # att mask
        attention_mask = torch.zeros(self.max_length)
        attention_mask[:len(tokens)] = 1

        token_type_ids = torch.zeros(self.max_length)

        # padding
        while len(tokens) < self.max_length:
            tokens.append('[PAD]')
            labels.append('O')

        # to ids
        input_ids  = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids  = torch.tensor(input_ids).long()

        mention_ids = list(map(lambda x: label2id[x], labels))
        mention_ids = torch.tensor(mention_ids).long()

        attention_mask = attention_mask.long()
        token_type_ids = token_type_ids.long()

        span_ids = torch.zeros(self.max_length, self.max_length)
        i = 0
        while i < len(labels):
            if labels[i] != "O":
                start = i
                event_type = labels[i]
                end = i
                while end < len(labels) and labels[end+1] == event_type:
                    end += 1
                i = end
                span_ids[start,end] = 1
                assert labels[start:end+1][0] == event_type
            i += 1

        return input_ids, attention_mask, token_type_ids, mention_ids, span_ids

    def enhance(self, support_samples, target_classes):
        enhance_samples = []
        enhance_samples.extend(support_samples)
        for support in support_samples:
            enhance = {"id":str(support["id"])+"_enhance","tokens": support["tokens"], "events":[]}
            for j in support["events"]:
                if j["event_type"] in target_classes:
                    enhance_event = j
                    trigger_len = j["end"] - j["start"]
                    trigger = j["text"]
                    trigger_tag = get_wordnet_pos(pos_tag([trigger])[0][-1])
                    lu_list = list(self.framenet[j["event_type"]].lexUnit.keys())
                    lu = None
                    for lexunit in lu_list:
                        word = lexunit.split('.')[0]
                        word_len = len(word.split(' '))
                        tag = lexunit.split('.')[1]
                        if word_len == trigger_len and tag == trigger_tag:
                            lu = word
                            break
                    if lu is None:
                        continue
                    # while lu is None:
                    #     sample_lu = random.sample(lu_list,1)[0]
                    #     word = sample_lu.split('.')[0]
                    #     word_len = len(word.split(' '))
                    #     tag = sample_lu.split('.')[1]
                    #     # if word_len == trigger_len and tag == trigger_tag:
                    #     if word_len == trigger_len:
                    #         lu = word
                    enhance_event["text"] = lu
                    enhance["events"].append(enhance_event)
            enhance_samples.append(enhance)
        return enhance_samples

    def __len__(self):
        return 99999999

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        # label2id, id2label = self.__build_dict__(target_classes)
        label2id = { j:i for i,j in enumerate(target_classes)}
        id2label = { i:j for i,j in enumerate(target_classes)}
        label2id['O'] = -100
        id2label[-100] = 'O'
        support_set = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'mention_ids': [], 'span_ids': [] }
        query_set = {'input_ids': [], 'attention_mask': [], 'token_type_ids': [], 'mention_ids': [], 'span_ids': [] }

        support_samples = self.__sample__(target_classes, self.K)
        enhance_samples = self.enhance(support_samples, target_classes)
        for sample in enhance_samples:
            instance = self.preprocess(sample, target_classes)
            # token_ids, label_ids, B_mask, I_mask, text_mask, att_mask = self.tokenize(instance, label2id)
            input_ids, attention_mask, token_type_ids, mention_ids, span_ids = self.tokenize(instance, label2id)
            support_set['input_ids'].append(input_ids)
            support_set['attention_mask'].append(attention_mask)
            support_set['token_type_ids'].append(token_type_ids)
            support_set['mention_ids'].append(mention_ids)
            support_set['span_ids'].append(span_ids)

        query_samples = self.__sample__(target_classes, self.Q)
        for sample in query_samples:
            instance = self.preprocess(sample, target_classes)
            input_ids, attention_mask, token_type_ids, mention_ids, span_ids = self.tokenize(instance, label2id)
            query_set['input_ids'].append(input_ids)
            query_set['attention_mask'].append(attention_mask)
            query_set['token_type_ids'].append(token_type_ids)
            query_set['mention_ids'].append(mention_ids)
            query_set['span_ids'].append(span_ids)           

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
                                        args.framenet,
                                        args.max_len, 
                                        args.N, args.K, args.Q)
    return dataset

