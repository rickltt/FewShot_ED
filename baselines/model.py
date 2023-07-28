import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel
from fscrf import FewShotCRF
from crf import CRF

class PACRF(BertPreTrainedModel): 
    def __init__(self, args):
        super(PACRF, self).__init__(args.encoder.config)
        self.encoder = args.encoder
        self.feature_size = self.encoder.config.hidden_size
        self.max_len = args.max_len
        self.gpu = args.device
        self.sample_num = 5
        self.drop = nn.Dropout(args.dropout_prob)

        self.crf = FewShotCRF(2*args.N + 1, batch_first=True)

        # self attention
        self.Wk = nn.Linear(self.feature_size, self.feature_size)
        self.Wq = nn.Linear(self.feature_size, self.feature_size)
        self.Wv = nn.Linear(self.feature_size, self.feature_size)
    
        # crf score
        self.W_start_mean = nn.Linear(self.feature_size, 1)
        self.W_start_log_var = nn.Linear(self.feature_size, 1)
            
        self.W_end_mean = nn.Linear(self.feature_size, 1)
        self.W_end_log_var = nn.Linear(self.feature_size, 1)
        
        self.W_trans_mean = nn.Linear(self.feature_size * 2, 1)
        self.W_trans_log_var = nn.Linear(self.feature_size * 2, 1)

    def proto_interaction(self, prototype):
         # self attention
        K = self.Wk(prototype)  # 2*N+1, feature_size
        Q = self.Wq(prototype)  # 2*N+1, feature_size
        V = self.Wv(prototype)  # 2*N+1, feature_size
        
        att_score = torch.matmul(K, Q.transpose(-1, -2))                # 2*N+1, 2*N+1
        att_score /= torch.sqrt(torch.tensor(self.feature_size).to(self.gpu))  # 2*N+1, 2*N+1
        att_score = att_score.softmax(-1)                               # 2*N+1, 2*N+1
        
        prototype = torch.matmul(att_score, V)  # 2*N+1, feature_size
        return prototype  

    def compute_trans_score(self, prototype):
        label_num = prototype.size(0)
        left_prototype = prototype.unsqueeze(0).expand(label_num, label_num, -1)
        right_prototype = prototype.unsqueeze(1).expand(label_num, label_num, -1)
        cat_prototype = torch.cat([left_prototype, right_prototype], dim=-1)
        
        trans_mean = self.W_trans_mean(cat_prototype).squeeze(-1)               
        trans_log_var = self.W_trans_log_var(cat_prototype).squeeze(-1)
        
        trans_score = self.sampling(trans_mean, trans_log_var)
        
        return trans_score
    
    def generate_transition_score(self, prototype):
        # calculate crf score
        start_mean = self.W_start_mean(prototype).squeeze(-1)   # 2*N+1
        start_log_var = self.W_start_log_var(prototype).squeeze(-1)   # 2*N+1
        start_score = self.sampling(start_mean, start_log_var)   # sample_num, 2*N+1
        
        end_mean = self.W_end_mean(prototype).squeeze(-1)       # 2*N+1
        end_log_var = self.W_end_log_var(prototype).squeeze(-1)       # 2*N+1
        end_score = self.sampling(end_mean, end_log_var)       # sample_num, 2*N+1
        
        # reparameterize
        trans_score = self.compute_trans_score(prototype) # sample_num, 2*N+1, 2*N+1
        
        return start_score, end_score, trans_score

    def sampling(self, mean, logvar):
        epsilon = torch.randn(self.sample_num, *mean.shape).to(self.gpu)
        samples = mean + torch.exp(0.5 * logvar) * epsilon
        return samples
    
    
    def get_transition_score(self, prototype):
        # self attention
        prototype = self.proto_interaction(prototype)
        prototype = self.drop(prototype.relu())
        
        # calculate crf score        
        start_score, end_score, trans_score = self.generate_transition_score(prototype)
        
        return start_score, end_score, trans_score
    
    def forward(self, support_set, query_set, N, mode):
        # encode
        support_emb = self.encoder(input_ids=support_set['input_ids'], attention_mask=support_set['attention_mask'], token_type_ids =support_set['token_type_ids'])[0]
        query_emb = self.encoder(input_ids=query_set['input_ids'], attention_mask=query_set['attention_mask'], token_type_ids =query_set['token_type_ids'])[0]
        support_set_label = support_set['label_ids']
        # dropout
        support_emb = self.drop(support_emb)                # support_num, max_len, feature_size
        query_emb = self.drop(query_emb)                    # query_num, max_len, feature_size
        
        # prototype, 2*N+1, feature_size
        prototype = torch.empty(2*N+1, self.feature_size).to(self.gpu)
        for i in range(2*N+1):
            indices = support_set_label.view(-1) == i
            if indices.any():
                features = support_emb.view(-1,self.feature_size)[indices]
                features = features.sum(0) / features.shape[0]
            else:
                features = torch.ones_like(prototype[0]).to(self.gpu)
            prototype[i] = features
        
        # crf score
        # sample_num x 2*N+1, sample_num x 2*N+1, sample_num x 2*N+1 x 2*N+1
        start_score, end_score, trans_score = self.get_transition_score(prototype)

        query_label = query_set['label_ids'] # query_num, max_len

        query_num = query_emb.size(0)
        tag_num = prototype.size(0)
        query_emb = query_emb.unsqueeze(-2) # query_num, max_len, 1, feature_size
        query_emb = query_emb.expand(-1, -1, tag_num, -1) # query_num, max_len, tag_num, feature_size
        prototype = prototype.unsqueeze(0) # 1, query_num, feature_size
        prototype = prototype.unsqueeze(1) # 1, 1, query_num, feature_size
        prototype = prototype.expand(query_num, self.max_len, -1, -1) # query_num, max_len, query_num, feature_size

        logits = (prototype * query_emb).sum(-1) # query_num, max_len, 2*N+1
    
        if mode == "train":
            loss = []
            query_label = torch.where(query_label >= 0, query_label, torch.zeros_like(query_label))
            for i in range(self.sample_num):
                self.crf.set_transitions(start_score[i], end_score[i], trans_score[i])
                t_loss = -self.crf(logits, query_label, reduction='mean')
                loss.append(t_loss)
            loss = sum(loss) / len(loss) if len(loss) != 0 else 0
            return loss
        else:
            start_score = start_score.mean(dim=0)
            end_score = end_score.mean(dim=0)
            trans_score = trans_score.mean(dim=0)
            self.crf.set_transitions(start_score, end_score, trans_score)
            pred = self.crf.decode(logits)
            pred = torch.tensor(pred).view(-1).to(self.device)
            return pred


class VanillaCRF(BertPreTrainedModel): 
    def __init__(self, args):
        super(VanillaCRF, self).__init__(args.encoder.config)
        self.encoder = args.encoder
        self.feature_size = self.encoder.config.hidden_size
        self.max_len = args.max_len
        self.gpu = args.device
        self.cost = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(args.dropout_prob)
        self.crf = CRF(num_tags=2*args.N + 1, batch_first=True)

    def forward(self, support_set, query_set, N, mode):
        # encode
        support_emb = self.encoder(input_ids=support_set['input_ids'], attention_mask=support_set['attention_mask'], token_type_ids =support_set['token_type_ids'])[0]
        query_emb = self.encoder(input_ids=query_set['input_ids'], attention_mask=query_set['attention_mask'], token_type_ids =query_set['token_type_ids'])[0]
        support_set_label = support_set['label_ids']
        # dropout
        support_emb = self.drop(support_emb)                # support_num, max_len, feature_size
        query_emb = self.drop(query_emb)                    # query_num, max_len, feature_size
        
        # prototype, 2*N+1, feature_size
        prototype = torch.empty(2*N+1, self.feature_size).to(self.gpu)
        for i in range(2*N+1):
            indices = support_set_label.view(-1) == i
            if indices.any():
                features = support_emb.view(-1,self.feature_size)[indices]
                features = features.sum(0) / features.shape[0]
            else:
                features = torch.ones_like(prototype[0]).to(self.gpu)
            prototype[i] = features
        
        query_label = query_set['label_ids'] # query_num, feature_size

        query_num = query_emb.size(0)
        tag_num = prototype.size(0)
        query_emb = query_emb.unsqueeze(-2) # query_num, max_len, 1, feature_size
        query_emb = query_emb.expand(-1, -1, tag_num, -1) # query_num, max_len, tag_num, feature_size
        prototype = prototype.unsqueeze(0) # 1, query_num, feature_size
        prototype = prototype.unsqueeze(1) # 1, 1, query_num, feature_size
        prototype = prototype.expand(query_num, self.max_len, -1, -1)

        logits = (prototype * query_emb).sum(-1)

        if mode == "train":
            query_label = torch.where(query_label >= 0, query_label, torch.zeros_like(query_label))
            loss = -self.crf(logits, query_label, mask=query_set['attention_mask'])
            return loss
        else:
            pred = self.crf.decode(logits)
            pred = torch.tensor(pred).view(-1).to(self.gpu)
            return pred
        
class UnifiedModel(BertPreTrainedModel): 
    def __init__(self, args):
        super(UnifiedModel, self).__init__(args.encoder.config)
        self.encoder = args.encoder
        self.feature_size = self.encoder.config.hidden_size
        self.max_len = args.max_len
        self.metric = args.metric
        self.gpu = args.device
        if self.metric == 'relation':
            self.score = nn.Sequential(
                nn.Linear(2*self.feature_size, self.feature_size),
                nn.ReLU(),
                nn.Dropout(args.dropout_prob),
                nn.Linear(self.feature_size, 1))
        self.cost = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(args.dropout_prob)

    def __dist__(self, x, y, dim):
        if self.metric == 'dot':
            return (x * y).sum(dim)
        elif self.metric == 'euclidean':
            return -(torch.pow(x - y, 2)).sum(dim)
        elif self.metric == 'cosine':
            return 1e3 * F.cosine_similarity(x, y, dim=dim)
        elif self.metric == 'relation':
            input_feature = torch.cat((x, y), dim=dim)   # B, N*Q, max_len, 2*N+1, 2*feature_size
            sim = self.score(input_feature).squeeze(-1)             # B, N*Q, max_len, 2*N+1
            return sim
        else:
            raise ValueError("please enter dot, euclidean or cosine!")
        
    def forward(self, support_set, query_set, N, mode):
        # encode
        support_emb = self.encoder(input_ids=support_set['input_ids'], attention_mask=support_set['attention_mask'], token_type_ids =support_set['token_type_ids'])[0]
        query_emb = self.encoder(input_ids=query_set['input_ids'], attention_mask=query_set['attention_mask'], token_type_ids =query_set['token_type_ids'])[0]
        support_set_label = support_set['label_ids']
        # dropout
        support_emb = self.drop(support_emb)                # support_num, max_len, feature_size
        query_emb = self.drop(query_emb)                    # query_num, max_len, feature_size
        
        # prototype, 2*N+1, feature_size
        prototype = torch.empty(2*N+1, self.feature_size).to(self.gpu)
        for i in range(2*N+1):
            indices = support_set_label.view(-1) == i
            if indices.any():
                features = support_emb.view(-1,self.feature_size)[indices]
                features = features.sum(0) / features.shape[0]
            else:
                features = torch.ones_like(prototype[0]).to(self.gpu)
            prototype[i] = features

        tag_num = prototype.shape[0]
        query_num = query_emb.shape[0]

        query_emb = query_emb.unsqueeze(-2) # query_num, max_len, 1, feature_size
        query_emb = query_emb.expand(-1, -1, tag_num, -1)  # query_num, max_len, tag_num, feature_size

        prototype = prototype.unsqueeze(0) # 1, tag_num, feature_size
        prototype = prototype.unsqueeze(1) # 1, 1, tag_num, feature_size
        prototype = prototype.expand(query_num, self.max_len, -1, -1) # query_num, max_len, tag_num, feature_size

        logits = self. __dist__(query_emb, prototype, -1)
        # loss
        if mode == 'train':
            logits = logits.view(-1, tag_num)
            labels = query_set['label_ids'].view(-1) 
            loss = self.cost(logits, labels)
            return loss
        else:
            pred = F.softmax(logits,dim=-1).argmax(-1) 
            return pred
        