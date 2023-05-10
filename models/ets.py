import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel

class ETS(BertPreTrainedModel): 
    def __init__(self, args):
        super(ETS, self).__init__(args.encoder.config)
        self.encoder = args.encoder
        self.feature_size = self.encoder.config.hidden_size
        self.max_len = args.max_len
        self.tokenizer = args.tokenizer
        self.cost = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(args.dropout_prob)

    def forward(self, support_set, query_set, id2label, N, K, Q, mode):
        # encode
        support_emb = self.encoder(input_ids=support_set['tokens'], attention_mask=support_set['att-mask'])[0]   # B*N*K, max_len, feature_size
        query_emb = self.encoder(input_ids=query_set['tokens'], attention_mask=query_set['att-mask'])[0]       # B*N*K, max_len, feature_size
        
        labeltype = self.process_label(id2label)  # B, 2*N+1, feature_size

        # dropout
        support_emb = self.drop(support_emb)                # B*N*K, max_len, feature_size
        query_emb = self.drop(query_emb)                    # B*N*K, max_len, feature_size
        
        support_emb = support_emb.view(-1, N, K, self.max_len, self.feature_size)   # B, N, K, max_len, feature_size
        query_emb = query_emb.view(-1, N*Q, self.max_len, self.feature_size)        # B, N*Q, max_len, feature_size

        B_mask = support_set['B-mask'].view(-1, N, K, self.max_len)     # B, N, K, max_len
        I_mask = support_set['I-mask'].view(-1, N, K, self.max_len)     # B, N, K, max_len
        text_mask = support_set['text-mask'].view(-1, N, K, self.max_len)     # B, N, K, max_len

        # prototype 
        prototype = self.proto(support_emb, B_mask, I_mask, text_mask)         # B, 2*N+1, feature_size
        
        B = support_emb.shape[0]
        label_loss = []
        for i in range(B):
            l_loss = F.cosine_embedding_loss(prototype[i], labeltype[i], torch.ones(prototype[i].shape[0]).to(support_emb))
            label_loss.append(l_loss)
        
        label_loss = sum(label_loss) / len(label_loss) if len(label_loss) != 0 else 0
        # classification
        logits = self.similarity(prototype, query_emb)              # B, N*Q, max_len, 2*N+1
               
        # loss
        if mode == 'train':
            logits = logits.view(-1, logits.shape[-1]) # B*N*Q*max_len, 2*N+1
            label = query_set['trigger_label'].view(-1) # B*N*Q*max_len
            loss = self.cost(logits, label) + label_loss
            return loss
        else:
            pred = logits.view(-1, logits.shape[-1]).argmax(-1) 
            return pred, support_emb.view(-1, self.feature_size)
        
    def proto(self, support_emb, B_mask, I_mask, text_mask):
        '''
        input:
            support_emb : B, N, K, max_len, feature_size
            B_mask : B, N, K, max_len
            I_mask : B, N, K, max_len
        output:
            prototype : B, 2*N+1, feature_size # (class_num -> 2N + 1)
        '''
        B, N, K, _, _ = support_emb.shape
        prototype = torch.empty(B, 2*N+1, self.feature_size).to(support_emb) # B, 2*N+1, feature_size
        
        B_mask = B_mask.unsqueeze(-1)
        B_mask = B_mask.expand(-1, -1, -1, -1, self.feature_size)
        B_mask = B_mask.to(support_emb) # B, N, K, max_len, feature_size
        I_mask = I_mask.unsqueeze(-1)
        I_mask = I_mask.expand(-1, -1, -1, -1, self.feature_size)
        I_mask = I_mask.to(support_emb) # B, N, K, max_len, feature_size

        text_mask = text_mask.unsqueeze(-1)
        text_mask = text_mask.expand(-1, -1, -1, -1, self.feature_size)
        text_mask = text_mask.to(support_emb) # B, N, K, max_len, feature_size

        for i in range(B):
            O_mask = torch.ones_like(B_mask[i]).to(B_mask)    # N, K, max_len, feature_size
            O_mask -= B_mask[i] + I_mask[i]
            O_mask = O_mask * text_mask[i]
            for j in range(N):
                sum_B_fea = (support_emb[i, j] * B_mask[i, j]).view(-1, self.feature_size).sum(0)
                num_B_fea = B_mask[i, j].sum() / self.feature_size + 1e-8
                prototype[i, 2*j+1] = sum_B_fea / num_B_fea
                
                sum_I_fea = (support_emb[i, j] * I_mask[i, j]).view(-1, self.feature_size).sum(0)
                num_I_fea = I_mask[i, j].sum() / self.feature_size + 1e-8
                prototype[i, 2*j+2] = sum_I_fea / num_I_fea
            
            sum_O_fea = (support_emb[i] * O_mask[i]).reshape(-1, self.feature_size).sum(0)
            num_O_fea = O_mask[i].sum() / self.feature_size + 1e-8
            prototype[i, 0] = sum_O_fea / num_O_fea
        
        return prototype
    
    def process_label(self, id2label):
        CLS = []
        for i in id2label:
            labels = list(i.values())
            labels.remove('PAD')
            events = []
            for l in labels:
                if l == 'O':
                    events.append(['other'])
                else:
                    if 'E-Mail' in l:
                        l = l.replace('E-Mail','EMail')
                    l = l.replace('_','-')
                    l = l.replace('.','-')
                    l = l.split('-')
                    ll = []
                    for t in l:
                        if t == 'B':
                            ll.append('begin')
                        elif t == 'I':
                            ll.append('inside')
                        elif t == 'Org':
                            ll.append('organization')
                        else:
                            ll.append(t.lower())
                    events.append(ll)
            token_ids = []
            attentions = []
            token_types = []
            for e in events:
                token = []
                for word in e:
                    word_token = self.tokenizer.tokenize(word)
                    token += word_token
                assert len(token) <= 8
                token = ['[CLS]'] + token + ['[SEP]']
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                # token_id = [101] + token_id + [102]
                attention = [1] * len(token_id)
                padding_length = 10 - len(token_id)
                token_id += [0] * padding_length
                attention += [0] * padding_length
                token_type = [0] * len(token_id)
                token_ids.append(token_id)
                attentions.append(attention)
                token_types.append(token_type)
            token_ids = torch.tensor(token_ids, dtype=torch.long).to(self.device)
            attentions = torch.tensor(attentions, dtype=torch.long).to(self.device)
            token_types = torch.tensor(token_types, dtype=torch.long).to(self.device)
            cls = self.encoder(input_ids=token_ids, attention_mask=attentions, token_type_ids=token_types)[1]
            CLS.append(cls)
        CLS = torch.stack(CLS)
        return CLS

    def similarity(self, prototype, query):
        '''
        inputs:
            prototype: B, 2*N+1, feature_size
            query: B, N*Q, max_len, feature_size
        outputs:
            sim: B, N*Q, max_len, 2*N+1
        '''
        tag_num = prototype.shape[1]  # 2*N+1
        query_num = query.shape[1] # N*Q
        
        query = query.unsqueeze(-2)                                 # B, N*Q, max_len, 1, feature_size
        query = query.expand(-1, -1, -1, tag_num, -1)               # B, N*Q, max_len, 2*N+1, feature_size
        
        prototype = prototype.unsqueeze(1)                          # B, 1, 2*N+1, feature_size
        prototype = prototype.unsqueeze(2)                          # B, 1, 1, 2*N+1, feature_size
        prototype = prototype.expand(-1, query_num, self.max_len, -1, -1) # B, N*Q, max_len, 2*N+1, feature_size
        
        sim = (prototype * query).sum(-1)                           # B, N*Q, max_len, 2*N+1

        return sim
 

