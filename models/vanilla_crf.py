import torch
from torch import nn
from .unified_model import UnifiedModel
from .fscrf import FewShotCRF

class VanillaCRF(UnifiedModel):
    def __init__(self, args):
        super(VanillaCRF, self).__init__(args)
        self.crf = FewShotCRF(2*args.evalN + 1, batch_first=True)
        self.crf.init_transitions()

        # self attention
        self.metric = 'dot'
        self.Wk = nn.Linear(self.feature_size, self.feature_size)
        self.Wq = nn.Linear(self.feature_size, self.feature_size)
        self.Wv = nn.Linear(self.feature_size, self.feature_size)
    
    def proto_interaction(self, prototype):
        # self attention
        K = self.Wk(prototype)  # B, 2*N+1, feature_size
        Q = self.Wq(prototype)  # B, 2*N+1, feature_size
        V = self.Wv(prototype)  # B, 2*N+1, feature_size
        
        att_score = torch.matmul(K, Q.transpose(-1, -2))                # B, 2*N+1, 2*N+1
        att_score /= torch.sqrt(torch.tensor(self.feature_size).to(K))  # B, 2*N+1, 2*N+1
        att_score = att_score.softmax(-1)                               # B, 2*N+1, 2*N+1
        
        prototype = torch.matmul(att_score, V)  # B, 2*N+1, feature_size
        return prototype
    
    def forward(self, support_set, query_set, N, K, Q, mode):
        # encode
        support_emb = self.encoder(input_ids=support_set['tokens'], attention_mask=support_set['att-mask'])[0]   # B*N*K, max_len, feature_size
        query_emb = self.encoder(input_ids=query_set['tokens'], attention_mask=query_set['att-mask'])[0]       # B*N*K, max_len, feature_size
        
        # dropout
        support_emb = self.drop(support_emb)                # B*N*K, max_len, feature_size
        query_emb = self.drop(query_emb)                    # B*N*K, max_len, feature_size
        
        support_emb = support_emb.view(-1, N, K, self.max_len, self.feature_size)   # B, N, K, max_len, feature_size
        query_emb = query_emb.view(-1, N*Q, self.max_len, self.feature_size)        # B, N*Q, max_len, feature_size

        B_mask = support_set['B-mask'].view(-1, N, K, self.max_len)     # B, N, K, max_len
        I_mask = support_set['I-mask'].view(-1, N, K, self.max_len)     # B, N, K, max_len

        # text_mask = support_set['text-mask'].view(-1, N, K, self.max_len) 
        # prototype 
        prototype = self.proto1(support_emb, B_mask, I_mask)         # B, 2*N+1, feature_size
    
        # classification
        B = prototype.shape[0]
        logits = []
        pred = []
        loss = []
        query_label = query_set['trigger_label'].view(B, N*Q, self.max_len)
        query_att = query_set['att-mask'].view(B, N*Q, self.max_len)

        for i in range(B):
            t_query_emb = query_emb[i].unsqueeze(0)  # 1, N*Q, max_len, feature_size
            t_prototype = prototype[i].unsqueeze(0)  # 1, 2*N+1, feature_size
            t_query_label = query_label[i]
            t_att_mask = query_att[i].long()

            t_logits = self.similarity(t_prototype, t_query_emb)  # 1, N*Q, max_len, 2*N+1
            t_logits = t_logits.view(-1, self.max_len, 2*N+1)
            t_query_label = torch.where(t_query_label >= 0, t_query_label, torch.zeros_like(t_query_label))
            
            logits.append(t_logits)
            t_loss = -self.crf(t_logits, t_query_label, t_att_mask)
            loss.append(t_loss)

            t_pred = self.crf.decode(t_logits, t_att_mask)
            pred.extend(t_pred.view(-1))        
            
        # loss
        if mode == 'train':
            loss = sum(loss) / len(loss) if len(loss) != 0 else 0
            return loss
        else:
            pred = torch.tensor(pred).to(query_emb.device)
            return pred, support_emb



