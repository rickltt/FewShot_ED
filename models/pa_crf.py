import torch
from torch import nn
from .unified_model import UnifiedModel
from .fscrf import FewShotCRF

class PACRF(UnifiedModel):
    def __init__(self, args):
        super(PACRF, self).__init__(args)
        self.crf = FewShotCRF(2*args.evalN + 1, batch_first=True)
        # self.sample_num = args.sample_num
        self.sample_num = 5

        # self attention
        self.metric = 'dot'
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
        K = self.Wk(prototype)  # B, 2*N+1, feature_size
        Q = self.Wq(prototype)  # B, 2*N+1, feature_size
        V = self.Wv(prototype)  # B, 2*N+1, feature_size
        
        att_score = torch.matmul(K, Q.transpose(-1, -2))                # B, 2*N+1, 2*N+1
        att_score /= torch.sqrt(torch.tensor(self.feature_size).to(K))  # B, 2*N+1, 2*N+1
        att_score = att_score.softmax(-1)                               # B, 2*N+1, 2*N+1
        
        prototype = torch.matmul(att_score, V)  # B, 2*N+1, feature_size
        return prototype
    
    def compute_trans_score(self, prototype):
        B, label_num, _ = prototype.shape
        left_prototype = prototype.unsqueeze(1).expand(-1, label_num, label_num, -1)
        right_prototype = prototype.unsqueeze(2).expand(-1, label_num, label_num, -1)
        cat_prototype = torch.cat([left_prototype, right_prototype], dim=-1)
        
        trans_mean = self.W_trans_mean(cat_prototype).squeeze(-1)             # B, 2*N+1, feature_size        
        trans_log_var = self.W_trans_log_var(cat_prototype).squeeze(-1)
        
        trans_score = self.sampling(trans_mean, trans_log_var)
        
        return trans_score
        
    def generate_transition_score(self, prototype):
        # calculate crf score
        start_mean = self.W_start_mean(prototype).squeeze(-1)   # B, 2*N+1
        start_log_var = self.W_start_log_var(prototype).squeeze(-1)   # B, 2*N+1
        start_score = self.sampling(start_mean, start_log_var)   # B, 2*N+1
        
        end_mean = self.W_end_mean(prototype).squeeze(-1)       # B, 2*N+1
        end_log_var = self.W_end_log_var(prototype).squeeze(-1)       # B, 2*N+1
        end_score = self.sampling(end_mean, end_log_var)       # B, 2*N+1
        
        # reparameterize
        trans_score = self.compute_trans_score(prototype)
        
        return start_score, end_score, trans_score
    
    def sampling(self, mean, logvar):
        epsilon = torch.randn(self.sample_num, *mean.shape).to(mean.device)
        samples = mean + torch.exp(0.5 * logvar) * epsilon
        return samples
        
    def get_transition_score(self, prototype):
        # self attention
        prototype = self.proto_interaction(prototype)
        prototype = self.drop(prototype.relu())
        
        # calculate crf score        
        start_score, end_score, trans_score = self.generate_transition_score(prototype)
        
        return start_score, end_score, trans_score
       
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
        
        # crf score
        start_score, end_score, trans_score = self.get_transition_score(prototype)
        
        # start_score,end_score sample_num x B x 2*N+1
        # trans_score sample_num x B x 2*N+1 x 2*N+1

        # classification
        B = prototype.shape[0]
        query_label = query_set['trigger_label'].view(B, N*Q, self.max_len)
        query_att = query_set['att-mask'].view(B, N*Q, self.max_len)
        logits = []
        pred = []
        loss = []
        for i in range(B):
            t_query_emb = query_emb[i].unsqueeze(0)  # 1, N*Q, max_len, feature_size
            t_prototype = prototype[i].unsqueeze(0)  # 1, 2*N+1, feature_size
            t_query_label = query_label[i]
            t_att_mask = query_att[i].long()

            t_start_score = start_score[:,i]  # 2*N+1
            t_end_score = end_score[:,i]      # 2*N+1
            t_trans_score = trans_score[:,i]  # 2*N+1, 2*N+1
            
            t_logits = self.similarity(t_prototype, t_query_emb)  # 1, N*Q, max_len, 2*N+1
            t_logits = t_logits.view(-1, self.max_len, 2*N+1)
            t_query_label = torch.where(t_query_label >= 0, t_query_label, torch.zeros_like(t_query_label))
            # t_logits = torch.tanh(t_logits)
            logits.append(t_logits)
            for j in range(self.sample_num):
                self.crf.set_transitions(t_start_score[j], 
                                             t_end_score[j], 
                                             t_trans_score[j])
                t_loss = -self.crf(t_logits, t_query_label, t_att_mask, reduction='mean')
                loss.append(t_loss)
  
            t_start_score = t_start_score.mean(dim=0)
            t_end_score = t_end_score.mean(dim=0)
            t_trans_score = t_trans_score.mean(dim=0)

            self.crf.set_transitions(t_start_score, 
                                         t_end_score, 
                                         t_trans_score)
            t_pred = self.crf.decode(t_logits, t_att_mask)
            pred.extend(t_pred.view(-1))
            # if self.training:                
            #     for j in range(self.sample_num):
            #         self.crf.set_transitions(t_start_score[j], 
            #                                  t_end_score[j], 
            #                                  t_trans_score[j])
            #         t_loss = -self.crf(t_logits, t_query_label, reduction='mean')
            #         loss.append(t_loss)
            # else:
            #     t_start_score = t_start_score.mean(dim=0)
            #     t_end_score = t_end_score.mean(dim=0)
            #     t_trans_score = t_trans_score.mean(dim=0)

            #     self.crf.set_transitions(t_start_score, 
            #                              t_end_score, 
            #                              t_trans_score)
                
            #     t_pred = self.crf.decode(t_logits)
            #     pred.append(t_pred)             
            
        # loss
        if mode == 'train':
            loss = sum(loss) / len(loss) if len(loss) != 0 else 0
            return loss
        else:
            pred = torch.tensor(pred).to(query_emb.device)
            return pred, support_emb



