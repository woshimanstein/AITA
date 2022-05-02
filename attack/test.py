import os
import argparse
import numpy as np
import torch

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM

# Attack
from attacker import attack_identity

# SRL
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

# setup args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--classifier',
    type=str,
    default='roberta-base',
    help=f'Model to attack'
)
arg_parser.add_argument(
    '--model_checkpoint',
    type=str,
    required=True,
    help=f'Path to model checkpoint'
)
arg_parser.add_argument(
    '--dataset',
    type=str,
    required=True,
    help=f'Name of the dataset for attack'
)
arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)
args = arg_parser.parse_args()

os.chdir('../')
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# language model
language_model = AutoModelForMaskedLM.from_pretrained('roberta-base')

# load classifier to be attacked
if args.classifier == 'roberta-base':
    classifier = AutoModelForSequenceClassification.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model_checkpoint = torch.load(args.model_checkpoint, map_location='cuda')
classifier.load_state_dict(model_checkpoint)

# load dataset
dataset = datasets.load_dataset('csv', data_files={'dev': os.path.join('data', 'aita', args.dataset)})

# Semantic role labeler
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

for data in dataset['dev']:
    input = data['title'] + data['body']

    if 'AITA for telling my stepdaughter to stop' in input:
        output = tokenizer(input, return_tensors='pt', truncation=True)

        # original prediction
        classifier_logits = classifier(output['input_ids']).logits
        print(classifier_logits)
        original_pred = classifier_logits.argmax(axis=-1).item()

        attack_identity(language_model, classifier, input, tokenizer, original_pred, 30, predictor)
