# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from bert_score.utils import get_idf_dict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import transformers
import math
import argparse
import math
import jiwer
import numpy as np
import os
import warnings
transformers.logging.set_verbosity(transformers.logging.ERROR)
import time 
import torch
import torch.nn.functional as F
import datasets

from attack_src.utils import bool_flag, get_output_file, print_args, load_gpt2_from_dict


def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)


def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R


def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def main(args):
    os.chdir('../')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

    pretrained = args.model.startswith('textattack')
    output_file = get_output_file(args, args.model, args.start_index, args.start_index + args.num_samples)
    output_file = os.path.join(args.adv_samples_folder, output_file)
    print(f"Outputting files to {output_file}")
    if os.path.exists(output_file):
        print('Skipping batch as it has already been completed.')
        exit()
    
    # Load dataset
    # DELETED: dataset, num_labels = load_data(args)
    train_data_file_name = os.path.join('data', 'aita', f'train_balanced_aita_custom_agr_0.9_comment_10.csv')
    dev_data_file_name = os.path.join('data', 'aita', f'dev_balanced_aita_custom_agr_0.9_comment_10.csv')
    dataset = datasets.load_dataset('csv', data_files={'train': train_data_file_name, 'dev': dev_data_file_name})
    num_labels = 2

    label_perm = lambda x: x

    # Load tokenizer, model, and reference model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = 512
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels).cuda()
    if not pretrained:
        # Load model to attack
        model_checkpoint = os.path.join('model_weights', 'bert-base-cased-aita_custom_class_only_0.9_10_balanced_title_selftext_bsz_6_seed_12', 'checkpoint-22555', 'pytorch_model.bin')
        print('Loading checkpoint: %s' % model_checkpoint)
        model.load_state_dict(torch.load(model_checkpoint))
        tokenizer.model_max_length = 512
        
    ref_model = load_gpt2_from_dict("%stransformer_wikitext-103.pth" % args.gpt2_checkpoint_folder, output_hidden_states=True).cuda()
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().cuda())
        
    # encode dataset using tokenizer
    testset_key = 'dev'
    def preprocess_function(batch):
        batch_input = []
        for b in range(len(batch['title'])):
            batch_input.append(batch['title'][b] + ' ' + batch['body'][b])
        tokenized_input = tokenizer(batch_input, padding=True, truncation=True)
        labels = {'labels': np.array(batch['is_asshole'])}
        return {**tokenized_input, **labels}
    encoded_dataset = dataset.map(preprocess_function, batched=True)
        
    # Compute idf dictionary for BERTScore
    if args.constraint == "bertscore_idf":
        idf_dict = get_idf_dict(dataset['train']['body'], tokenizer, nthreads=20)

    adv_log_coeffs, clean_texts, adv_texts = [], [], []

    clean_logits = []
    adv_logits = []
    token_errors = []
    times = []
    
    assert args.start_index < len(encoded_dataset[testset_key]), 'Starting index %d is larger than dataset length %d' % (args.start_index, len(encoded_dataset[testset_key]))
    end_index = min(args.start_index + args.num_samples, len(encoded_dataset[testset_key]))
    adv_losses, ref_losses, perp_losses, entropies = torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters), torch.zeros(end_index - args.start_index, args.num_iters)
    for idx in range(args.start_index, end_index):
        input_ids = encoded_dataset[testset_key]['input_ids'][idx]
        token_type_ids = None #encoded_dataset[testset_key]['token_type_ids'][idx]
        label = label_perm(encoded_dataset[testset_key]['labels'][idx])
        clean_logit = model(input_ids=torch.LongTensor(input_ids).unsqueeze(0).cuda(),
                             token_type_ids=(None if token_type_ids is None else torch.LongTensor(token_type_ids).unsqueeze(0).cuda())).logits.data.cuda()
        print('LABEL')
        print(label)
        print('TEXT')
        print(tokenizer.decode(input_ids))
        print('LOGITS')
        print(clean_logit)
        
        forbidden = np.zeros(len(input_ids)).astype('bool')
        # set [CLS] and [SEP] tokens to forbidden
        forbidden[0] = True
        forbidden[-1] = True
        offset = 0 if args.model == 'gpt2' else 1
        forbidden_indices = np.arange(0, len(input_ids))[forbidden]
        forbidden_indices = torch.from_numpy(forbidden_indices).cuda()
        token_type_ids_batch = (None if token_type_ids is None else torch.LongTensor(token_type_ids).unsqueeze(0).repeat(args.batch_size, 1).cuda())

        start_time = time.time()
        with torch.no_grad():
            orig_output = ref_model(torch.LongTensor(input_ids).cuda().unsqueeze(0)).hidden_states[args.embed_layer]
            if args.constraint.startswith('bertscore'):
                if args.constraint == "bertscore_idf":
                    ref_weights = torch.FloatTensor([idf_dict[idx] for idx in input_ids]).cuda()
                    ref_weights /= ref_weights.sum()
                else:
                    ref_weights = None
            elif args.constraint == 'cosine':
                # GPT-2 reference model uses last token embedding instead of pooling
                if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                    orig_output = orig_output[:, -1]
                else:
                    orig_output = orig_output.mean(1)
            log_coeffs = torch.zeros(len(input_ids), embeddings.size(0))
            indices = torch.arange(log_coeffs.size(0)).long()
            log_coeffs[indices, torch.LongTensor(input_ids)] = args.initial_coeff
            log_coeffs = log_coeffs.cuda()
            log_coeffs.requires_grad = True

        optimizer = torch.optim.Adam([log_coeffs], lr=args.lr)
        start = time.time()
        for i in range(args.num_iters):
            optimizer.zero_grad()
            coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(args.batch_size, 1, 1), hard=False) # B x T x V
            inputs_embeds = (coeffs @ embeddings[None, :, :]) # B x T x D
            pred = model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch).logits
            if args.adv_loss == 'ce':
                adv_loss = -F.cross_entropy(pred, label * torch.ones(args.batch_size).long().cuda())
            elif args.adv_loss == 'cw':
                top_preds = pred.sort(descending=True)[1]
                correct = (top_preds[:, 0] == label).long()
                indices = top_preds.gather(1, correct.view(-1, 1))
                adv_loss = (pred[:, label] - pred.gather(1, indices) + args.kappa).clamp(min=0).mean()
            
            # Similarity constraint
            ref_embeds = (coeffs @ ref_embeddings[None, :, :])
            pred = ref_model(inputs_embeds=ref_embeds)
            if args.lam_sim > 0:
                output = pred.hidden_states[args.embed_layer]
                if args.constraint.startswith('bertscore'):
                    ref_loss = -args.lam_sim * bert_score(orig_output, output, weights=ref_weights).mean()
                else:
                    if args.model == 'gpt2' or 'bert-base-uncased' in args.model:
                        output = output[:, -1]
                    else:
                        output = output.mean(1)
                    cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
                    ref_loss = -args.lam_sim * cosine.mean()
            else:
                ref_loss = torch.Tensor([0]).cuda()
               
            # (log) perplexity constraint
            if args.lam_perp > 0:
                perp_loss = args.lam_perp * log_perplexity(pred.logits, coeffs)
            else:
                perp_loss = torch.Tensor([0]).cuda()
                
            # Compute loss and backward
            total_loss = adv_loss + ref_loss + perp_loss
            total_loss.backward()
            
            entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
            if i % args.print_every == 0:
                print('Iteration %d: loss = %.4f, adv_loss = %.4f, ref_loss = %.4f, perp_loss = %.4f, entropy=%.4f, time=%.2f' % (
                    i+1, total_loss.item(), adv_loss.item(), ref_loss.item(), perp_loss.item(), entropy.item(), time.time() - start))

            # Gradient step
            log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()

            # Log statistics
            adv_losses[idx - args.start_index, i] = adv_loss.detach().item()
            ref_losses[idx - args.start_index, i] = ref_loss.detach().item()
            perp_losses[idx - args.start_index, i] = perp_loss.detach().item()
            entropies[idx - args.start_index, i] = entropy.detach().item()
        times.append(time.time() - start_time)
        
        print('CLEAN TEXT')
        clean_text = tokenizer.decode(input_ids[offset:(len(input_ids)-offset)])
        clean_texts.append(clean_text)
        print(clean_text)
        clean_logits.append(clean_logit)
        
        print('ADVERSARIAL TEXT')
        with torch.no_grad():
            for j in range(args.gumbel_samples):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)
                adv_ids = adv_ids[offset:len(adv_ids)-offset].to('cpu').tolist()
                adv_text = tokenizer.decode(adv_ids)
                x = tokenizer(adv_text, max_length=256, truncation=True, return_tensors='pt')
                token_errors.append(wer(adv_ids, x['input_ids'][0]))
                adv_logit = model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                                  token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None)).logits.data.to('cpu')
                if adv_logit.argmax() != label or j == args.gumbel_samples - 1:
                    adv_texts.append(adv_text)
                    print(adv_text)
                    adv_logits.append(adv_logit)
                    break
                
        # remove special tokens from adv_log_coeffs
        adv_log_coeffs.append(log_coeffs[offset:(log_coeffs.size(0)-offset), :].to('cpu')) # size T x V
        
        print('')
        print('CLEAN LOGITS')
        print(clean_logit) # size 1 x C
        print('ADVERSARIAL LOGITS')
        print(adv_logit)   # size 1 x C
            
    print("Token Error Rate: %.4f (over %d tokens)" % (sum(token_errors) / len(token_errors), len(token_errors)))
    torch.save({
        'adv_log_coeffs': adv_log_coeffs, 
        'adv_logits': torch.cat(adv_logits, 0), # size N x C
        'adv_losses': adv_losses,
        'adv_texts': adv_texts,
        'clean_logits': torch.cat(clean_logits, 0), 
        'clean_texts': clean_texts, 
        'entropies': entropies,
        'labels': list(map(label_perm, encoded_dataset[testset_key]['labels'][args.start_index:end_index])), 
        'perp_losses': perp_losses,
        'ref_losses': ref_losses,
        'times': times,
        'token_error': token_errors,
    }, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="White-box attack.")

    # Bookkeeping
    parser.add_argument("--result_folder", default="result/", type=str,
        help="folder for loading trained models")
    parser.add_argument("--gpt2_checkpoint_folder", default="result/", type=str,
        help="folder for loading GPT2 model trained with BERT tokenizer")
    parser.add_argument("--adv_samples_folder", default="adv_samples/", type=str,
        help="folder for saving generated samples")
    parser.add_argument("--dump_path", default="", type=str,
        help="Path to dump logs")

    # Data
    parser.add_argument("--dataset", default="dbpedia14", type=str,
        choices=["dbpedia14", "ag_news", "imdb", "yelp", "mnli"],
        help="classification dataset to use")
    parser.add_argument("--num_samples", default=1, type=int,
        help="number of samples to attack")

    # Model
    parser.add_argument("--model", default="bert-base-cased", type=str,
        help="type of model")
    parser.add_argument("--finetune", default=False, type=bool_flag,
        help="load finetuned model")

    # Attack setting
    parser.add_argument("--start_index", default=0, type=int,
        help="starting sample index")
    parser.add_argument("--num_iters", default=100, type=int,
        help="number of epochs to train for")
    parser.add_argument("--batch_size", default=3, type=int,
        help="batch size for gumbel-softmax samples")
    parser.add_argument("--initial_coeff", default=15, type=int,
        help="initial log coefficients")
    parser.add_argument("--adv_loss", default="cw", type=str,
        choices=["cw", "ce"],
        help="adversarial loss")
    parser.add_argument("--constraint", default="bertscore_idf", type=str,
        choices=["cosine", "bertscore", "bertscore_idf"],
        help="constraint function")
    parser.add_argument("--lr", default=3e-1, type=float,
        help="learning rate")
    parser.add_argument("--kappa", default=5, type=float,
        help="CW loss margin")
    parser.add_argument("--embed_layer", default=-1, type=int,
        help="which layer of LM to extract embeddings from")
    parser.add_argument("--lam_sim", default=1, type=float,
        help="embedding similarity regularizer")
    parser.add_argument("--lam_perp", default=1, type=float,
        help="(log) perplexity regularizer")
    parser.add_argument("--print_every", default=10, type=int,
        help="print loss every x iterations")
    parser.add_argument("--gumbel_samples", default=100, type=int,
        help="number of gumbel samples; if 0, use argmax")

    args = parser.parse_args()
    print_args(args)
    main(args)