import os
import argparse
import numpy as np
import torch

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import datasets

import textattack
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import BAEGarg2019, BERTAttackLi2020
from textattack.shared import AttackedText

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
    '-b', '--batch',
    type=int,
    default=4,
    help=f'Specify batch size'
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
args = arg_parser.parse_args()

os.chdir('../')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

model = AutoModelForSequenceClassification.from_pretrained(args.model).cuda()
checkpoint_path = '/data/ziyuan/aita_pre_model_weights_april_4/roberta-base-aita_custom_class_only_0.9_10_balanced_title_selftext_bsz_6_seed_12/checkpoint-22555/pytorch_model.bin'
model_checkpoint = torch.load(checkpoint_path, map_location='cuda')
model.load_state_dict(model_checkpoint)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

attack = BAEGarg2019.build(model_wrapper)

if args.balanced:
    if args.control_agreement:
        dev_data_file_name = os.path.join('data', 'aita', f'dev_balanced_aita_custom_agr_{args.agreement}_comment_{args.num_comments}_B.csv')
    else:
        dev_data_file_name = os.path.join('data', 'aita', f'dev_balanced_aita_custom_agr_{args.agreement}_comment_{args.num_comments}.csv')
else:
    dev_data_file_name = os.path.join('data', 'aita', f'dev_aita_custom_agr_{args.agreement}_comment_{args.num_comments}.csv')
dataset = datasets.load_dataset('csv', data_files={'dev': dev_data_file_name})

def process_data(batch):
    batch_input = []
    for b in range(len(batch['title'])):
        batch_input.append(batch['title'][b] + ' ' + batch['body'][b])
    return {**{'text': batch_input}, **{'labels': np.array(batch['is_asshole'])}}

processed_dataset = dataset.map(
    process_data,
    batched=True,
    remove_columns=['title', 'body', 'comment', 'is_asshole']
    )

tuple_dataset = []
for i in range(len(processed_dataset['dev'])):
    tuple_dataset.append((processed_dataset['dev'][i]['text'], processed_dataset['dev'][i]['labels']))

attack_args = textattack.AttackArgs(
    num_examples=10,
    log_to_csv="attack_log.csv",
    checkpoint_interval=5,
    checkpoint_dir="attack_checkpoints",
    disable_stdout=True
)
attack_dataset = textattack.datasets.Dataset(dataset=tuple_dataset)
attacker = textattack.Attacker(attack, attack_dataset, attack_args)
attack_results = attacker.attack_dataset()