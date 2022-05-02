import os
import argparse
import sys
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import datasets
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

def main(args):
    '''
    Hyper-parameter 
    '''
    MODEL = args.model
    DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
    BATCH_SIZE = args.batch
    NUM_EPOCH = args.epoch
    SEED = args.seed
    '''
    Control and logging
    '''
    os.chdir('../')
    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICE_ID)
    # control randomness
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # model saving and logging paths
    os.makedirs(os.path.dirname('model_weights' + '/'), exist_ok=True)
    # model name
    MODEL_NAME = f'{MODEL}-social-chem_bsz_{BATCH_SIZE}_seed_{SEED}'


    '''
    Dataset
    '''
    train_data_file_name = os.path.join('data', 'social_chem', 'train_social_chem.csv')
    dev_data_file_name = os.path.join('data', 'social_chem', 'dev_social_chem.csv')
    dataset = datasets.load_dataset('csv', data_files={'train': train_data_file_name, 'dev': dev_data_file_name})

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def process_data(batch):
        tokenized_input = tokenizer(batch['action'], padding='max_length', truncation=True)
        labels_list = []
        for label in batch['action-moral-judgment']:
            label = int(label)
            if label < 0:
                labels_list.append(0)
            elif label > 0:
                labels_list.append(1)
            else:
                print(label)
        labels = {'labels': labels_list}
        return {**tokenized_input, **labels}

    processed_dataset = dataset.map(
        process_data,
        batched=True
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
        precision = precision_metric.compute(predictions=predictions, references=labels, average='micro')
        recall = recall_metric.compute(predictions=predictions, references=labels, average='micro')
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall}


    '''
    Trainer
    '''
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    training_args = TrainingArguments(
        output_dir=os.path.join('model_weights', MODEL_NAME),
        num_train_epochs=NUM_EPOCH,
        learning_rate=1e-5,
        weight_decay=0.001,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        seed = args.seed
        )
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=processed_dataset['train'], 
        eval_dataset=processed_dataset['dev'],
        compute_metrics=compute_metrics
        )
    trainer.train()

if __name__ == '__main__':
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
        default=2,
        help=f'Specify batch size'
    )
    arg_parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help=f'Specify random seed'
    )

    args = arg_parser.parse_args()
    main(args)