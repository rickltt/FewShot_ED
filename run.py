
import torch
import numpy as np
import random
import os
import argparse
from tqdm import trange
from torch.utils.data import DataLoader
from data_utils import load_dataset, collate_fn
from metric import Metric
from models import UnifiedModel, PACRF, VanillaCRF, ETS
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE

import logging
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, model, train_dataloader, dev_dataloader):
    """ Train the model """

    train_epoch = args.train_epoch
    warmup_step = args.warmup_step
    eval_step = args.eval_step
    eval_epoch = args.eval_epoch

    trainN = args.trainN
    K = args.K
    Q = args.Q

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in parameters_to_optimize  if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in parameters_to_optimize  if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps= train_epoch)
    best_f1 = 0
    train_iterator = trange(args.train_epoch, desc="Epoch")
    for epoch in train_iterator:    
        # train
        model.train()
        support_set, query_set, id2label = next(train_dataloader)
        for k in support_set.keys():
            support_set[k] = support_set[k].to(args.device)
        for k in query_set.keys():
            query_set[k] = query_set[k].to(args.device)

        if args.model == 'ets':
            loss = model(support_set, query_set, id2label, trainN, K, Q, 'train')
        else:
            loss = model(support_set, query_set, trainN, K, Q, 'train')
        
        loss = loss.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        train_iterator.set_description('Loss: {}'.format(round(loss.item(), 4)))

        # evaluate
        if (epoch+1) % eval_step == 0:
            p, r, f1 = evaluate(args, model, dev_dataloader, eval_epoch, draw=False)
    
            print(f"Evaluate result of epoch {epoch+1} - Precision : {p:.6f}, Recall : {r:.6f}, F1 : {f1:.6f}")
            if f1 > best_f1:
                best_f1 = f1
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model.save_pretrained(output_dir)
                # torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                args.tokenizer.save_pretrained(output_dir)
                # logger.info("Saving model checkpoint to %s", output_dir)
                print(f"New best performance in epoch {epoch+1} - Precision: {p:.6f}, Recall: {r:.6f}, F1: {f1:.6f}")


def evaluate(args, model, eval_dataloader, eval_epoch, draw):

    evalN = args.evalN
    K = args.K
    Q = args.Q

    model.eval() 
    metric = Metric()
    eval_iterator = trange(eval_epoch, desc="Evaluating")
    with torch.no_grad():
        for idx in eval_iterator:
            support_set, query_set, id2label = next(eval_dataloader)
            for k in support_set.keys():
                support_set[k] = support_set[k].to(args.device)
            for k in query_set.keys():
                query_set[k] = query_set[k].to(args.device)
            if args.model == 'ets':
                pred, support_emb = model(support_set, query_set, id2label, evalN, K, Q, 'test')
            else:
                pred, support_emb = model(support_set, query_set, evalN, K, Q, 'test')
            if draw and (idx+1) % 500 == 0 :
                support_labels = support_set["trigger_label"].view(-1).cpu().detach().numpy()
                support_features = support_emb.cpu().detach().numpy()
                if not os.path.exists('./figures'):
                    os.mkdir('./figures')
                draw_embeddings(support_features, support_labels, 
                                     os.path.join('./figures', args.model + "_" + str(idx) +  ".svg"), id2label[0], evalN)
            metric.update_state(pred, query_set['trigger_label'], id2label)
    return metric.result()

def draw_embeddings(support_features, support_labels, figure_path, id2label, N):
    tsne = TSNE(n_components=2, perplexity=3)

    labels = []
    for k,v in id2label.items():
        if k > 0:
            label = v[2:]
            if label not in labels:
                labels.append(label)
    print(labels)

    no_pad_support_labels = support_labels[np.where(support_labels > 0)]
    no_pad_support_features = support_features[np.where(support_labels > 0)]
    no_pad_support_labels = (no_pad_support_labels - 1) // 2

    X = no_pad_support_features
    y = no_pad_support_labels
    tsne_drawing_features = tsne.fit_transform(X)
    
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(1, 1, 1)
    classes_index = list(range(N))
    colors = ["red", "darkorange", "cornflowerblue", "royalblue", "y"]
    for key in classes_index:
        support_index = np.where(y == key)

        """
        ax.scatter(tsne_prototype_features[key][0], tsne_prototype_features[key][1],
                   marker='.', color=cmap(classes_index.index(key)), s=400, alpha=1.0)
        """
        ax.scatter(tsne_drawing_features[support_index][:, 0], tsne_drawing_features[support_index][:, 1],
                   marker='.', color=colors[key], label=key, s=120, alpha=1.0)
        """
        ax.scatter(tsne_query_features[query_index][:, 0], tsne_query_features[query_index][:, 1],
                   marker='.', color=cmap(classes_index.index(key)), label=key, s=100, alpha=1.0)
        """
        """
        ax.scatter(correct_tsne_query_features[correct_query_index][:, 0],
                       correct_tsne_query_features[correct_query_index][:, 1], marker='^',
                       color=cmap(classes_index.index(key)))
        ax.scatter(incorrect_tsne_query_features[incorrect_query_index][:, 0],
                       incorrect_tsne_query_features[incorrect_query_index][:, 1], marker='x',
                       color=cmap(classes_index.index(key)))
        """
    ax.legend(labels=labels, fontsize=5, loc="lower right")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(figure_path, format="svg", bbox_inches="tight")
    plt.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, 
                        help='seed')
    parser.add_argument('--data_dir', default='./data/maven', type=str, 
                        help='fewevent, ace, maven')
    parser.add_argument("--model_name_or_path", default='../bert/bert-base-uncased', type=str, help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--model', default='unified', type=str, 
                        help='unified, vanilla, pacrf, ets')
    parser.add_argument('--metric', default='dot', type=str, 
                    help='cosine, euclidean, dot, relation')

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument('--max_len', default=128, type=int, 
                        help='max sentence length')
    parser.add_argument('--trainN', default=5, type=int, 
                        help='train N')
    parser.add_argument('--evalN', default=5, type=int, 
                        help='eval N')
    parser.add_argument('--K', default=5, type=int, 
                        help="K")
    parser.add_argument('--Q', default=1, type=int, 
                        help="Q")
    parser.add_argument('--split', default=True, type=bool, 
                        help='split dataset')
    parser.add_argument('--batch_size', default=1, type=int, 
                        help='batch size')
    parser.add_argument('--dropout_prob', default=0.1, type=float, 
                        help='dropout rate')
    parser.add_argument('--learning_rate', default=1e-5, type=float, 
                        help='learnint rate')
    parser.add_argument('--warmup_step', default=100, type=int, 
                        help='warmup step of bert')
    parser.add_argument('--scheduler_step', default=1000, type=int, 
                        help='scheduler step')  
    parser.add_argument('--train_epoch', default=20000, type=int, 
                        help='train epoch')
    parser.add_argument('--eval_epoch', default=1000, type=int, 
                        help='eval epoch')
    parser.add_argument('--eval_step', default=500, type=int, 
                        help='eval step')
    parser.add_argument('--test_epoch', default=3000, type=int, 
                        help='test epoch') 
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    model_dict = {
        'pacrf': PACRF,
        'unified': UnifiedModel,
        'vanilla': VanillaCRF,
        'ets': ETS
    }

    # Set seed
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    encoder = AutoModel.from_pretrained(args.model_name_or_path)
    args.encoder = encoder
    args.tokenizer = tokenizer

    # load dataset    
    dataset = load_dataset(args)
    train_dataloader = DataLoader(dataset=dataset["train"],
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dataset["dev"],
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=dataset["test"],
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=collate_fn)
    
    train_dataloader = iter(train_dataloader)
    dev_dataloader = iter(dev_dataloader)
    test_dataloader = iter(test_dataloader)

    # train_dataloader = get_loader(args, 'train')
    # dev_dataloader = get_loader(args, 'dev')
    # test_dataloader = get_loader(args, 'test')

    model = model_dict[args.model](args)
    model.to(args.device)

    if args.do_train:
        # Training
        train(args, model, train_dataloader, dev_dataloader)

        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        # torch.save(model.state_dict(), os.path.join(output_dir, "model"))
        args.tokenizer.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

    if args.do_eval:
        # test
        checkpoint = os.path.join(args.output_dir, 'last_checkpoint')
        state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
        model.load_state_dict(state_dict)
        model.to(args.device)

        P, R, F1 = evaluate(args, model, test_dataloader, args.test_epoch, draw=True)
        dataset_name = args.data_dir.split('/')[-1]
        if args.model == 'unified':
            args.model = args.metric
        result = '***** Model: {} , Dataset:{}, N:{}, K:{},  Predict in test dataset *****\n'.format(args.model, dataset_name, args.trainN, args.K)
        result += f"Test result - Precision : {P:.6f}, Recall : {R:.6f}, F1 : {F1:.6f}"
        with open(os.path.join(args.output_dir,'test_result.txt'),'a') as f:
            f.write(result + '\n')
        print(result)

        
if __name__ == '__main__':
    main()