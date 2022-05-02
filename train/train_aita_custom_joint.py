import os
import argparse
import sys
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import datasets
from transformers import BartConfig, BartTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import TrainingArguments

from custom_model.bart_multi_task import BartForJointGenerationClassification
from train_utils import TrainerWithTrainingLoss
# setup args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--gpu1',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)
arg_parser.add_argument(
    '--gpu2',
    type=int,
    default=0,
    help=f'Specify another gpu to use. If only one is needed, choose this to be the same as gpu1.'
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
    '--gradient_acc',
    type=int,
    default=12,
    help=f'Gradient accumulation steps'
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
arg_parser.add_argument(
    '--num_comments',
    type=int,
    default=10
)
arg_parser.add_argument(
    '--agreement',
    type=float,
    default=0.9
)
arg_parser.add_argument(
    '--control_agreement',
    action='store_true'
)
arg_parser.add_argument(
    '--alpha',
    type=float,
    default=0.7
)
args = arg_parser.parse_args()

os.chdir('../')

'''
Hyper-parameter 
'''
if args.gpu1 == args.gpu2:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu1)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu1) + ',' + str(args.gpu2)
BATCH_SIZE = args.batch
GRADIENT_ACCUMULATION = args.gradient_acc
NUM_EPOCH = args.epoch
SEED = args.seed
LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
AGREEMENT = args.agreement
LEAST_NUM_COMMENTS = args.num_comments
'''
Control and logging
'''
# control randomness
torch.manual_seed(SEED)
np.random.seed(SEED)
# model saving and logging paths
os.makedirs(os.path.dirname('/data/ziyuan/model_weights/'), exist_ok=True)
# model name
MODEL_NAME = f'bart-base-aita_custom'
if args.balanced:
    MODEL_NAME += '_balanced'
if args.title:
    MODEL_NAME += '_title'
if args.selftext:
    MODEL_NAME += '_selftext'
MODEL_NAME += f'{args.alpha }_bsz_{BATCH_SIZE}_seed_{SEED}'
MODEL_NAME = MODEL_NAME.replace('/', '-')


'''
Dataset
'''
if args.balanced:
    if args.control_agreement:
        train_data_file_name = os.path.join('data', 'aita', f'train_balanced_aita_custom_agr_{AGREEMENT}_comment_{LEAST_NUM_COMMENTS}_B.csv')
        dev_data_file_name = os.path.join('data', 'aita', f'dev_balanced_aita_custom_agr_{AGREEMENT}_comment_{LEAST_NUM_COMMENTS}_B.csv')
    else:
        train_data_file_name = os.path.join('data', 'aita', f'train_balanced_aita_custom_agr_{AGREEMENT}_comment_{LEAST_NUM_COMMENTS}.csv')
        dev_data_file_name = os.path.join('data', 'aita', f'dev_balanced_aita_custom_agr_{AGREEMENT}_comment_{LEAST_NUM_COMMENTS}.csv')
else:
    train_data_file_name = os.path.join('data', 'aita', f'train_aita_custom_agr_{AGREEMENT}_comment_{LEAST_NUM_COMMENTS}.csv')
    dev_data_file_name = os.path.join('data', 'aita', f'dev_aita_custom_agr_{AGREEMENT}_comment_{LEAST_NUM_COMMENTS}.csv')

# train_data_file_name = os.path.join('data', 'aita', 'mini_aita_custom.csv')
# dev_data_file_name = os.path.join('data', 'aita', 'mini_aita_custom.csv')
dataset = datasets.load_dataset('csv', data_files={'train': train_data_file_name, 'dev': dev_data_file_name})

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

def process_data(batch):
    if args.title and (not args.selftext):
        tokenized_input = tokenizer(batch['title'], padding=False, truncation=True, max_length=512)
    elif (not args.title) and args.selftext:
        tokenized_input = tokenizer(batch['body'], padding=False, truncation=True, max_length=512)
    elif args.title and args.selftext:
        batch_input = []
        for b in range(len(batch['title'])):
            batch_input.append(batch['title'][b] + ' ' + batch['body'][b])
        tokenized_input = tokenizer(batch_input, padding=False, truncation=True, max_length=512)
    else:
        raise RuntimeError('You have to select at least one from title and selftext.')
    
    labels = {'labels': tokenizer(batch['comment'], padding=False, truncation=True, max_length=512)['input_ids']}
    classification_target = {'classification_target': batch['is_asshole']}

    return {**tokenized_input, **labels, **classification_target}

processed_dataset = dataset.map(
    process_data,
    batched=True,
    remove_columns=['title', 'body', 'comment', 'is_asshole']
    )


'''
Metrics
'''
accuracy_metric = datasets.load_metric('accuracy')
precision_metric = datasets.load_metric('precision')
recall_metric = datasets.load_metric('recall')

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    logits = logits[0]
    labels = labels[1]
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(predictions=predictions, references=labels)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}


'''
Trainer
'''

model = BartForJointGenerationClassification.from_pretrained("facebook/bart-base")

data_collator = DataCollatorForSeq2Seq(tokenizer, model, padding='longest')

training_args = TrainingArguments(
    output_dir=os.path.join('/', 'data', 'ziyuan', 'model_weights', MODEL_NAME),
    num_train_epochs=NUM_EPOCH,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    eval_accumulation_steps=10,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    label_names=['labels', 'classification_target']
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
trainer.train()