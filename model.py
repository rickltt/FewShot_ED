import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel


class SpanFSED(nn.Module): 
    def __init__(self, args):
        super(SpanFSED, self).__init__()
        self.span_detector = SpanDetector(args)
        self.gpu = args.device
        self.lam = 0.5
        self.mention_classifier = MentionClassifier(args)

    def forward(self, support_set, query_set, N, mode):

        if mode == "train":
            span_loss, _ = self.span_detector(support_set)
            mention_loss, _ = self.mention_classifier(support_set, query_set, N, query_set["mention_ids"])
            total_loss = self.lam * span_loss + (1-self.lam) * mention_loss
            return total_loss
        else:
            _, span_pred = self.span_detector(query_set)
            span_label = torch.full_like(query_set["mention_ids"],-100).to(self.gpu)
            for i in span_pred:
                span_label[i[0]][i[1]:i[2]+1] = 1
            _, preds = self.mention_classifier(support_set, query_set, N, span_label)
            return preds

class MentionClassifier(nn.Module): 
    def __init__(self, args):
        super(MentionClassifier, self).__init__()
        self.gpu = args.device
        self.encoder = BertModel.from_pretrained(args.model_name_or_path)
        self.feature_size = self.encoder.config.hidden_size
        self.cost = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(args.dropout_prob)

    def proto(self, N, emb, label):
        prototype = torch.empty(N, self.feature_size).to(self.gpu)
        for i in range(N):
            indices = label.view(-1) == i
            features = emb.view(-1,self.feature_size)[indices]
            features = features.sum(0) / features.shape[0]
            prototype[i] = features
        return prototype
    
    def __dist__(self, prototype, query_emb):
        tag_num = prototype.shape[0]
        query_num = query_emb.shape[0]
        query_emb = query_emb.unsqueeze(-2) # query_num, max_len, 1, feature_size
        query_emb = query_emb.expand(-1, tag_num, -1)  # query_num, max_len, tag_num, feature_size
        prototype = prototype.unsqueeze(0) # 1, tag_num, feature_size
        prototype = prototype.expand(query_num, -1, -1) # query_num, max_len, tag_num, feature_size
        mention_logits = -(torch.pow(prototype - query_emb, 2)).sum(-1)

        return mention_logits
    
    def forward(self, support_set, query_set, N, span_label):
        # encode
        support_emb = self.encoder(input_ids=support_set['input_ids'], attention_mask=support_set['attention_mask'], token_type_ids =support_set['token_type_ids'])[0]
        # dropout
        support_emb = self.drop(support_emb)                # support_num, max_len, feature_size

        support_set_label = support_set['mention_ids']
        prototype = self.proto(N, support_emb, support_set_label)

        query_emb = self.encoder(input_ids=query_set['input_ids'], attention_mask=query_set['attention_mask'], token_type_ids =query_set['token_type_ids'])[0]
        query_emb = self.drop(query_emb)

        span_indices = span_label.view(-1) != -100
        query_emb = query_emb.view(-1,self.feature_size)[span_indices]
        mention_logits = self.__dist__(prototype, query_emb)

        query_label = query_set["mention_ids"]
        query_mention_label = query_label.view(-1)[span_indices]
        mention_loss = self.cost(mention_logits, query_mention_label)

        pred = F.softmax(mention_logits,dim=-1).argmax(-1) 

        prediction = torch.full_like(query_set["mention_ids"], -100).to(self.gpu)
        prediction = prediction.view(-1)
        prediction[span_indices] = pred

        return mention_loss, prediction

class SpanDetector(nn.Module): 
    def __init__(self, args):
        super(SpanDetector, self).__init__()
        self.encoder = BertModel.from_pretrained(args.model_name_or_path)
        self.gpu = args.device

        self.inner_dim = 64
        self.hidden_size = self.encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.inner_dim * 2)

        self.RoPE = True

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.gpu)
        return embeddings
    
    def multilabel_categorical_crossentropy(self, y_true, y_pred):
        """
        https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()
    
    def loss_fun(self, y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = self.multilabel_categorical_crossentropy(y_true, y_pred)
        return loss
    
    def forward(self, inputs):

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        span_ids = inputs["span_ids"]
        last_hidden_state = self.encoder(input_ids, attention_mask, token_type_ids)[0]

        batch_size, seq_len = last_hidden_state.shape[0], last_hidden_state.shape[1]

        # outputs:(batch_size, seq_len, inner_dim*2)
        outputs = self.dense(last_hidden_state)

        # qw,kw:(batch_size, seq_len, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, inner_dim)
            cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, seq_len, seq_len)
        logits = torch.einsum('bmd,bnd->bmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12
        logits = logits / self.inner_dim ** 0.5
        loss = self.loss_fun(span_ids,logits)

        logits = logits.detach().cpu().numpy()
        pred = []
        for b, start, end in zip(*np.where(logits > 0)):
            pred.append((b, start, end))

        # span_ids = span_ids.detach().cpu().numpy()
        # true = []
        # for b, start, end in zip(*np.where(span_ids > 0)):
        #     true.append((b, start, end))
        return loss, pred
    
