import torch
import numpy as np
import random
import os
import argparse
from tqdm import trange
from torch.utils.data import DataLoader
from utils import collate_fn, load_dataset
from model import SpanFSED
from metric import Metric
from sentence_transformer import get_framenet
from transformers import (
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import logging
logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, model, train_dataloader, dev_dataloader):
    """ Train the model """

    train_epoch = args.train_epoch
    eval_step = args.eval_step
    eval_epoch = args.eval_epoch
    trainN = args.N

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in parameters_to_optimize  if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in parameters_to_optimize  if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    warmup_step = int(0.1*train_epoch)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps= train_epoch)
    best_f1 = 0
    train_iterator = trange(args.train_epoch, desc="Epoch")
    for epoch in train_iterator:    
        # train
        model.train()
        support_set, query_set, _ = next(train_dataloader)
        for k in support_set.keys():
            support_set[k] = support_set[k].to(args.device)
        for k in query_set.keys():
            query_set[k] = query_set[k].to(args.device)

        loss = model(support_set, query_set, trainN, 'train')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        train_iterator.set_description('Loss: {}'.format(round(loss.item(), 4)))

        # evaluate
        if (epoch+1) % eval_step == 0:
            f1 = evaluate(args, model, dev_dataloader, eval_epoch)
            logger.info(f"Evaluate result of epoch {epoch+1} - F1 : {f1:.6f}")
            if f1 > best_f1:
                best_f1 = f1
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                torch.save(model.state_dict(),os.path.join(output_dir, "model"))
                # model.save_pretrained(output_dir)
                args.tokenizer.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
                logger.info(f"New best performance in epoch {epoch+1} - F1: {f1:.6f}")

def evaluate(args, model, eval_dataloader, eval_epoch):

    evalN = args.N
    model.eval() 
    metric = Metric()
    eval_iterator = trange(eval_epoch, desc="Evaluating")
    with torch.no_grad():
        for _ in eval_iterator:
            support_set, query_set, id2label = next(eval_dataloader)
            for k in support_set.keys():
                support_set[k] = support_set[k].to(args.device)
            for k in query_set.keys():
                query_set[k] = query_set[k].to(args.device)

            pred = model(support_set, query_set, evalN, 'test')
            metric.update_state(pred, query_set["mention_ids"], id2label)
    return metric.result()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, 
                        help='seed')
    parser.add_argument('--data_dir', default='./data/ere', type=str)
    parser.add_argument("--model_name_or_path", default='../bert/bert-base-uncased', type=str, help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='output', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--max_len', default=128, type=int, 
                        help='max sentence length')
    parser.add_argument('--N', default=5, type=int, 
                        help='train N')
    parser.add_argument('--K', default=5, type=int, 
                        help="K")
    parser.add_argument('--Q', default=1, type=int, 
                        help="Q")
    parser.add_argument('--dropout_prob', default=0.1, type=float, 
                        help='dropout rate')
    parser.add_argument('--learning_rate', default=3e-5, type=float, 
                        help='learnint rate')
    parser.add_argument('--train_epoch', default=200, type=int, 
                        help='train epoch')
    parser.add_argument('--eval_epoch', default=50, type=int, 
                        help='eval epoch')
    parser.add_argument('--eval_step', default=50, type=int, 
                        help='eval step')
    parser.add_argument('--test_epoch', default=500, type=int, 
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

    # Set seed
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer

    dataset_name = args.data_dir.split('/')[-1]
    framenet = get_framenet(dataset_name)
    args.framenet = framenet
    # load dataset    
    dataset = load_dataset(args)
    train_dataloader = DataLoader(dataset=dataset["train"],
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dataset["dev"],
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=dataset["test"],
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=collate_fn)
    
    train_dataloader = iter(train_dataloader)
    dev_dataloader = iter(dev_dataloader)
    test_dataloader = iter(test_dataloader)

    model = SpanFSED(args)
    model.to(args.device)

    if args.do_train:
        # Training
        train(args, model, train_dataloader, dev_dataloader)
        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(),os.path.join(output_dir, "model"))
        args.tokenizer.save_pretrained(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)

    if args.do_eval:
        # test
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)
        model.to(args.device)

        F1 = evaluate(args, model, test_dataloader, args.test_epoch)
    
        result = '***** Model: SpanFSED , Dataset:{}, N:{}, K:{}, seed:{} *****\n'.format(dataset_name, args.N, args.K, args.seed)
        result += f"Test result - F1 : {F1:.6f}"
        with open(os.path.join(args.output_dir,'test_result.txt'),'a') as f:
            f.write(result + '\n')
        logger.info(result)
       
if __name__ == '__main__':
    main()