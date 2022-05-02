import os
import tqdm
import argparse
import sys
sys.path.insert(0, os.path.abspath('..'))

from dataset.dataset import SocialIQaDataset

import torch
import numpy as np
from transformers import AutoModelForMultipleChoice, AutoTokenizer
from transformers import AdamW
import datasets

# setup args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--model',
    type=str,
    default='roberta-base'
)
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
    default=1,
    help=f'Specify batch size'
)
arg_parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help=f'Specify random seed'
)
args = arg_parser.parse_args()

os.chdir('../')

'''
hyper-parameter 
'''
MODEL = args.model
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
MODEL_NAME = f'{MODEL}-socialiqa_bsz_{BATCH_SIZE}_seed_{SEED}'
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.log'), 'w')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
model = AutoModelForMultipleChoice.from_pretrained(MODEL).to(device)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)



# record these for every epoch
loss_record = []

print(f'Training {MODEL} for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')

for epo in range(NUM_EPOCH):
    model.train()
    total_loss = 0
    
    '''
    DataLoader
    '''
    dataset = SocialIQaDataset(data='train')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # training
    train_iterator_with_progress = tqdm.tqdm(data_loader)
    idx = 0
    for batch in train_iterator_with_progress:
        # prepare input
        batch_answerA = batch['answerA']
        batch_answerB = batch['answerB']
        batch_answerC = batch['answerC']
        batch_label = (batch['label'] - 1).to(device)
        batch_prompt = []
        for b in range(BATCH_SIZE):
            batch_prompt.append(batch['context'][b] + '[UNUSED]' + batch['question'][b])

        # input encoding
        tokenizer.add_tokens(['[UNUSED]'])
        model.resize_token_embeddings(len(tokenizer))
        input_encoding = tokenizer(batch_prompt * 3, batch_answerA + batch_answerB + batch_answerC, return_tensors='pt', padding=True).to(device)

        # zero-out gradient
        optimizer.zero_grad()
        outputs = model(**{k: v.unsqueeze(0) for k,v in input_encoding.items()}, labels=batch_label)

        # compute loss and perform a step
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        idx += 1

        total_loss += float(loss)
        train_iterator_with_progress.set_description(f'Epoch {epo}')
        train_iterator_with_progress.set_postfix({'Loss': loss.item()})

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
        valid_dataset = SocialIQaDataset(data='dev')
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        accuracy = datasets.load_metric('accuracy')

        for batch in valid_data_loader:
            # prepare input
            batch_answerA = batch['answerA']
            batch_answerB = batch['answerB']
            batch_answerC = batch['answerC']
            batch_label = torch.tensor(batch['label']).to(device) - 1
            batch_prompt = []
            for b in range(BATCH_SIZE):
                batch_prompt.append(batch['context'][b] + '[UNUSED]' + batch['question'][b])

            # input encoding
            tokenizer.add_tokens(['[UNUSED]'])
            model.resize_token_embeddings(len(tokenizer))
            input_encoding = tokenizer(batch_prompt * 3, batch_answerA + batch_answerB + batch_answerC, return_tensors='pt', padding=True).to(device)

            # forward
            outputs = model(**{k: v.unsqueeze(0) for k,v in input_encoding.items()}, labels=batch_label)
            predictions = torch.argmax(outputs.logits, dim=1)

            accuracy.add_batch(predictions=predictions, references=batch_label)

        acc = accuracy.compute()
        print(f'acc in epoch {epo}: {acc}')
        log_file.write(f'acc:{acc} ')

    SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}_epoch_{epo+1}.pt')
    torch.save(model.state_dict(), SAVE_PATH)