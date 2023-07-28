from seqeval.metrics import f1_score
class Metric:
    def __init__(self):        
        self.pred = []
        self.true = []

    def convertBIO(self, labels):
        i = 0
        while i < len(labels):
            if labels[i] != 'O':
                event_type = labels[i]
                start = i
                end = i
                while end < len(labels) and labels[end+1] == event_type:
                    end += 1
                for j,k in enumerate(range(start,end+1)):
                    if j == 0:
                        labels[k] = "B-" + labels[k]
                    else:
                        labels[k] = "I-" + labels[k]
                i = end
            i += 1
        return labels
    
    def update_state(self, preds, trues, id2label):

        preds = preds.view(-1).cpu().tolist()
        trues = trues.view(-1).cpu().tolist()
        
        id2label[-100] = 'O'
        preds = [id2label[pred] for pred in preds]
        trues = [id2label[true] for true in trues]

        trues = self.convertBIO(trues)
        preds = self.convertBIO(preds)

        self.pred.append(preds)
        self.true.append(trues) 
    
    def result(self):
        f1 = f1_score(self.true, self.pred)
        return f1
    
