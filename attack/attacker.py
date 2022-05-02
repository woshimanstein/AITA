import copy
import transformers
import nltk
from sentence_splitter import split_text_into_sentences


def pos_tag_from_tokenized(tokenizer, tokenized_ids):
    tokens = []
    for j in range(tokenized_ids.shape[1]):
        tokens.append(tokenizer.decode(tokenized_ids[0, j]))
    token_pos = nltk.pos_tag(tokens)
    return token_pos

def attack_identity(
    langauge_model,
    classifier,
    text_attakced,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    original_label,
    top_k,
    srl):

    words_to_replace = set()
    sentences = split_text_into_sentences(text=text_attakced, language='en')
    for sentence in sentences:
        if len(sentence.strip()) != 0:
            srl_result = srl.predict(sentence)
            for verb in srl_result['verbs']:
                # print(verb['description'])
                description_list = verb['description'].split()
                combined_description_list = []
                curr_combined_constituent = ''
                for i in range(1, len(description_list)):
                    if '[' in description_list[i]:
                        curr_combined_constituent = description_list[i]
                    elif curr_combined_constituent != '' and ']' not in description_list[i]:
                        curr_combined_constituent = curr_combined_constituent + ' ' + description_list[i]
                    elif curr_combined_constituent != '' and ']' in description_list[i]:
                        curr_combined_constituent = curr_combined_constituent + ' ' + description_list[i]
                        combined_description_list.append(curr_combined_constituent)
                        curr_combined_constituent = ''
                for constituent in combined_description_list:
                    if constituent[1: -1].split(':')[0].strip() == 'ARG0':
                        words_to_replace.add(constituent[1: -1].split(':')[1].strip())

    result = bert_attack(langauge_model, classifier, text_attakced, tokenizer, words_to_replace, original_label, top_k)
    if result != None:
        with open('identity_attack.txt', 'a') as f:
            f.write(text_attakced + '\n')
            f.write('-' * 20 + '\n')
            f.write(result + '\n')
            f.write('-' * 20 + '\n')
            f.write('-' * 20 + '\n')

def attack_location(
    langauge_model,
    classifier,
    text_attakced,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    original_label,
    top_k,
    srl):

    words_to_replace = set()
    sentences = split_text_into_sentences(text=text_attakced, language='en')
    for sentence in sentences:
        if len(sentence.strip()) != 0:
            srl_result = srl.predict(sentence)
            for verb in srl_result['verbs']:
                # print(verb['description'])
                description_list = verb['description'].split()
                combined_description_list = []
                curr_combined_constituent = ''
                for i in range(1, len(description_list)):
                    if '[' in description_list[i]:
                        curr_combined_constituent = description_list[i]
                    elif curr_combined_constituent != '' and ']' not in description_list[i]:
                        curr_combined_constituent = curr_combined_constituent + ' ' + description_list[i]
                    elif curr_combined_constituent != '' and ']' in description_list[i]:
                        curr_combined_constituent = curr_combined_constituent + ' ' + description_list[i]
                        combined_description_list.append(curr_combined_constituent)
                        curr_combined_constituent = ''
                for constituent in combined_description_list:
                    if constituent[1: -1].split(':')[0].strip() == 'ARG-LOC':
                        words_to_replace.add(constituent[1: -1].split(':')[1].strip())
    print(words_to_replace)

    # result = bert_attack(langauge_model, classifier, text_attakced, tokenizer, words_to_replace, original_label, top_k)
    # if result != None:
    #     with open('location_attack.txt', 'a') as f:
    #         f.write(text_attakced + '\n')
    #         f.write('-' * 20 + '\n')
    #         f.write(result + '\n')
    #         f.write('-' * 20 + '\n')
    #         f.write('-' * 20 + '\n')

def replace_k(text, old, new, k):
    count = 0
    for i in range(len(text) - len(old) + 1):
        if text[i: i + len(old)] == old:
            count += 1
        if count == k:
            return text[: i] + new + text[i + len(old): ]
    return text


def bert_attack(
    langauge_model,
    classifier,
    text_attakced,
    tokenizer: transformers.tokenization_utils.PreTrainedTokenizer,
    words_to_replace,
    original_label,
    top_k):

    text_attakced = tokenizer.decode(tokenizer(text_attakced, return_tensors='pt', truncation=True)['input_ids'][0], skip_special_tokens=True)
    original_text_attakced = text_attakced
    for to_replace in words_to_replace:
        # POS of the original text
        # token_pos = pos_tag_from_tokenized(tokenizer, tokenized_ids)
        # original_pos = text_attakced

        # replacement
        to_replace = ' ' + to_replace + ' '
        text_attakced = original_text_attakced.replace(to_replace, '<mask>').replace('<mask>', ' <mask> ')
        
        tokenized_ids = tokenizer(text_attakced, return_tensors='pt', truncation=True)['input_ids']
        mask_token_ids = (tokenized_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        # query language model
        lm_logits = langauge_model(input_ids=tokenized_ids).logits
        
        # candidates
        candidate_ids = set()
        for i in range(mask_token_ids.shape[0]):
            mask_token_index = mask_token_ids[i].item()
            rank_ids = (-lm_logits[0, mask_token_index]).argsort(axis=-1)[0: top_k]
            for j in range(len(rank_ids)):
                candidate_ids.add(rank_ids[j])

        for idx in list(candidate_ids)[:100]:
            tokenized_ids_copy = copy.copy(tokenized_ids)
            tokenized_ids_copy[0, mask_token_ids] = idx
            classifier_logits = classifier(tokenized_ids_copy).logits
            predicted_class_id = classifier_logits.argmax(axis=-1).item()
            if predicted_class_id != original_label:
                print('success')
                print(classifier_logits)
                print(tokenizer.decode(tokenized_ids_copy[0], skip_special_tokens=True))
                return tokenizer.decode(tokenized_ids_copy[0], skip_special_tokens=True)


    print('fail')