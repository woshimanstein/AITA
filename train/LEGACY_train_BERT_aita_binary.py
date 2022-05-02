import os
import tqdm
import argparse
import sys
sys.path.insert(0, os.path.abspath('..'))

from dataset.dataset import BinaryClassificationDataset

import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AdamW
import datasets

# setup args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)
arg_parser.add_argument(
    '-e', '--epoch',
    type=int,
    default=5,
    help=f'Specify number of training epochs'
)
arg_parser.add_argument(
    '-b', '--batch',
    type=int,
    default=2,
    help=f'Specify batch size'
)
arg_parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help=f'Specify random seed'
)
arg_parser.add_argument(
    '--title',
    action='store_true',
    help='Use title as input'
)
arg_parser.add_argument(
    '--selftext',
    action='store_true',
    help='Use selftext as input'
)
arg_parser.add_argument(
    '--balanced',
    action='store_true',
    help='train on the balanced dataset'
)
arg_parser.add_argument(
    '--pretrained',
    action='store_true',
    help=f'Start from pretrained (SocialIQa, ...) model'
)
arg_parser.add_argument(
    '--pretrained_epoch',
    type=int,
    default=5,
    help=f'Specify number of epochs of pretraining'
)
args = arg_parser.parse_args()

os.chdir('../')

'''
hyper-parameter 
'''
DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
BATCH_SIZE = args.batch
NUM_EPOCH = args.epoch
SEED = args.seed
'''
control and logging
'''
# control randomness
torch.manual_seed(SEED)
np.random.seed(SEED)
# model saving and logging paths
os.makedirs(os.path.dirname('model_weights' + '/'), exist_ok=True)
MODEL_NAME = 'bert-base-aita_binary'
if args.balanced:
    MODEL_NAME += '_balanced'
if args.title:
    MODEL_NAME += '_title'
if args.selftext:
    MODEL_NAME += '_selftext'
if args.pretrained:
    MODEL_NAME += f'_socialiqa_{args.pretrained_epoch}'
MODEL_NAME += f'_bsz_{BATCH_SIZE}_seed_{SEED}'
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.log'), 'w')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
if args.pretrained:
    tokenizer.add_tokens(['[UNUSED]'])
    model.resize_token_embeddings(len(tokenizer))
    pretrained_model_path = os.path.join('model_weights', f'bert-base-socialiqa_bsz_1_seed_0_epoch_{args.pretrained_epoch}.pt')
    pretrained_checkpoint = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(dict([(n, p) for n, p in pretrained_checkpoint.items() if n not in ['classifier.weight', 'classifier.bias']]), strict=False)

# optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)



# record these for every epoch
loss_record = []

print(f'Training BART base for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')
for epo in range(NUM_EPOCH):
    model.train()
    total_loss = 0
    
    '''
    DataLoader
    '''
    dataset = BinaryClassificationDataset(data='train', balanced=args.balanced)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # training
    train_iterator_with_progress = tqdm.tqdm(data_loader)
    idx = 0
    for batch in train_iterator_with_progress:
        try:
            if args.title and (not args.selftext):
                input_encoding = tokenizer(batch['title'], return_tensors='pt', padding=True, truncation=True).to(device)
            elif (not args.title) and args.selftext:
                input_encoding = tokenizer(batch['body'], return_tensors='pt', padding=True, truncation=True).to(device)
            
            batch_target = batch['isasshole'].to(device)

            # input encoding
            input_encoding = tokenizer(batch_body, return_tensors='pt', padding=True, truncation=True).to(device)

            # zero-out gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(**input_encoding, labels=batch_target)

            # compute loss and perform a step
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            idx += 1

            total_loss += float(loss)
            train_iterator_with_progress.set_description(f'Epoch {epo}')
            train_iterator_with_progress.set_postfix({'Loss': loss.item()})
        except Exception as e:
            print(e)
    
    loss_record.append(total_loss)
    print(f'Loss in epoch {epo}: {total_loss}')
    log_file.write(f'Epoch:{epo} ')
    log_file.write(f'Loss:{total_loss} ')

    # evaluation
    model.eval()
    with torch.no_grad():
        '''
        DataLoader
        '''
        valid_dataset = BinaryClassificationDataset(data='dev', balanced=args.balanced)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

        accuracy = datasets.load_metric('accuracy')
        precision = datasets.load_metric('precision')
        recall = datasets.load_metric('recall')
        for batch in valid_data_loader:
            try:
                if args.title:
                    batch_body = batch['title']
                else:
                    batch_body = batch['body']
                
                batch_target = batch['isasshole'].to(device)

                # input encoding
                input_encoding = tokenizer(batch_body, return_tensors='pt', padding=True, truncation=True).to(device)

                # forward pass
                outputs = model(**input_encoding, labels=batch_target)
                predictions = torch.argmax(outputs.logits, dim=1)

                # accuracy
                accuracy.add_batch(predictions=predictions, references=batch_target)
                precision.add_batch(predictions=predictions, references=batch_target)
                recall.add_batch(predictions=predictions, references=batch_target)
            except Exception as e:
                print(e)
        
        acc = accuracy.compute()
        precision = precision.compute()
        recall = recall.compute()
        print(f'acc in epoch {epo}: {acc}')
        print(f'precision in epoch {epo}: {precision}')
        print(f'recall in epoch {epo}: {recall}')
        log_file.write(f'acc:{acc} ')
        log_file.write(f'precision:{precision} ')
        log_file.write(f'recall:{recall}\n')
    
    SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}_epoch_{epo+1}.pt')
    # save model after training for one epoch
    torch.save(model.state_dict(), SAVE_PATH)