import os
import argparse
import sys
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import pandas as pd
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments

from train_utils import TrainerWithTrainingLoss
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
    default=4,
    help=f'Specify batch size'
)
arg_parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help=f'Specify random seed'
)
arg_parser.add_argument(
    '--lr',
    type=float,
    default=1e-5,
    help=f'Learning rate'
)
arg_parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-3,
    help=f'Weight decay'
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
args = arg_parser.parse_args()

os.chdir('../')

'''
Hyper-parameter 
'''
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
MODEL = args.model
BATCH_SIZE = args.batch
NUM_EPOCH = args.epoch
SEED = args.seed
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
'''
Control and logging
'''
# control randomness
torch.manual_seed(SEED)
np.random.seed(SEED)
# model saving and logging paths
os.makedirs(os.path.dirname('model_weights' + '/'), exist_ok=True)
# model name
MODEL_NAME = f'{MODEL}-aita_scruples_class_only'
if args.balanced:
    MODEL_NAME += '_balanced'
if args.title:
    MODEL_NAME += '_title'
if args.selftext:
    MODEL_NAME += '_selftext'
MODEL_NAME += f'_bsz_{BATCH_SIZE}_seed_{SEED}'


'''
Dataset
'''
if args.balanced:
    train_data_file_name = os.path.join('data', 'aita', 'train.scruples-anecdotes.jsonl')
    dev_data_file_name = os.path.join('data', 'aita', 'dev.scruples-anecdotes.jsonl')
else:
    train_data_file_name = os.path.join('data', 'aita', 'train_aita_custom.csv')
    dev_data_file_name = os.path.join('data', 'aita', 'dev_aita_custom.csv')
# dataset = datasets.load_dataset('csv', data_files={'train': os.path.join('data', 'aita', 'mini_aita_clean.csv'), 'dev': os.path.join('data', 'aita', 'mini_aita_clean.csv')})
dataset = datasets.load_dataset('json', data_files={'train': train_data_file_name, 'dev': dev_data_file_name})

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.model_max_length = min(1024, tokenizer.model_max_length)

def process_data(batch):
    if args.title and (not args.selftext):
        tokenized_input = tokenizer(batch['title'], padding=False, truncation=True)
    elif (not args.title) and args.selftext:
        tokenized_input = tokenizer(batch['text'], padding=False, truncation=True)
    else:
        batch_input = []
        for b in range(len(batch['title'])):
            batch_input.append(batch['title'][b] + ' ' + batch['text'][b])
        tokenized_input = tokenizer(batch_input, padding=False, truncation=True)
    batch_labels = [1 if label == 'WRONG' else 0 for label in np.array(batch['binarized_label'])]
    labels = {'labels': batch_labels}
    return {**tokenized_input, **labels}

processed_dataset = dataset.map(
    process_data,
    batched=True,
    remove_columns=['index', 'id', 'post_id', 'action', 'title', 'text', 'post_type', 'label_scores', 'label', 'binarized_label_scores', 'binarized_label']
    )


'''
Metrics
'''
accuracy_metric = datasets.load_metric('accuracy')
precision_metric = datasets.load_metric('precision')
recall_metric = datasets.load_metric('recall')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}


'''
Trainer
'''
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
model.classifier_dropout = 0.2

data_collator = DataCollatorWithPadding(tokenizer, padding='longest')

training_args = TrainingArguments(
    output_dir=os.path.join('model_weights', MODEL_NAME),
    num_train_epochs=NUM_EPOCH,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    )
trainer = TrainerWithTrainingLoss(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=processed_dataset['train'], 
    eval_dataset=processed_dataset['dev'],
    compute_metrics=compute_metrics
    )


# test_data_file_name = os.path.join('data', 'aita', 'dev_balanced_aita_clean.csv')
# test_dataset = datasets.load_dataset('csv', data_files={'test': test_data_file_name})
# test_processed_dataset = test_dataset.map(
#     process_data,
#     batched=True,
#     remove_columns=['id', 'timestamp', 'title', 'body', 'edited', 'verdict', 'score', 'num_comments', 'is_asshole']
#     )
# eval_result = trainer.evaluate(test_processed_dataset['test'])
# print(eval_result)


trainer.train()

# eval_result = trainer.evaluate(test_processed_dataset['test'])
# print(eval_result)