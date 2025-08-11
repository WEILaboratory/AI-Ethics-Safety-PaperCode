import argparse
import regex as re
import nltk
import torch
import torch.multiprocessing as mp
from transformers import BertTokenizer, RobertaTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    tp = lambda x:list(x.split(','))

    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--neutral_words', type=tp, required=True)
    parser.add_argument('--attribute_words', type=tp, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'roberta'])
    parser.add_argument('--ab_test_type', type=str, default='final')
    args = parser.parse_args()

    return args

def prepare_tokenizer(args):
    if args.model_type == 'bert':
        pretrained_weights = 'bert-large-uncased'
        # pretrained_weights = 'pretrained_models/bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    elif args.model_type == 'roberta':
        pretrained_weights = 'roberta-large'
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
    return tokenizer


args = parse_args()
data = [l.strip() for l in open(args.input)]

# neutrals = [word.strip() for word in open(args.neutral_words)]
neutrals = []
for neutral in args.neutral_words:
    neutrals += [word.strip() for word in open(neutral)]
neutral_set = set(neutrals)

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

sequential_l = []
attributes_l = []
all_attributes_set = set()
for attribute in args.attribute_words:
    l = [word.strip() for word in open(attribute)]
    sequential_l.append(l)
    attributes_l.append(set(l))
    all_attributes_set |= set(l)

tokenizer = prepare_tokenizer(args)


def process(orig_line):
    res_list = []
    res = {
        'ori_label': None,
        'attributes_examples': (None, None),
        'attributes_labels': None,
        'ori_label_neutral': None,
        'neutral_examples': None,
        'neutral_labels': None,
    }
    neutral_flag = True
    orig_line = orig_line.strip()
    if len(orig_line) < 1:
        return res_list
    leng = len(orig_line.split())
    if leng > args.block_size or leng <= 1:
        return res_list
    tokens_orig = [token.strip() for token in re.findall(pat, orig_line)]
    tokens_lower = [token.lower() for token in tokens_orig]
    token_set = set(tokens_lower)

    for i, attribute_set in enumerate(attributes_l):
        if attribute_set & token_set:
            neutral_flag = False
            line = tokenizer.encode(orig_line, add_special_tokens=True)
            labels = attribute_set & token_set
            for ori_label in list(labels):
                idx = tokens_lower.index(ori_label)
                label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                line_ngram = list(nltk.ngrams(line, len(label)))
                try:
                    idx = line_ngram.index(label)
                    res['ori_label'] = ori_label
                    res['attributes_examples'] = (i, line)
                    res['attributes_labels'] = [idx + j for j in range(len(label))]
                    res_list.append(res)
                    res = {
                        'ori_label': None,
                        'attributes_examples': (None, None),
                        'attributes_labels': None,
                        'ori_label_neutral': None,
                        'neutral_examples': None,
                        'neutral_labels': None,
                    }
                except:
                    pass

    if neutral_flag:
        if neutral_set & token_set:
            line = tokenizer.encode(orig_line, add_special_tokens=True)
            labels = neutral_set & token_set
            for ori_label in list(labels):
                idx = tokens_lower.index(ori_label)
                label = tuple(tokenizer.encode(tokens_orig[idx], add_special_tokens=True))[1:-1]
                line_ngram = list(nltk.ngrams(line, len(label)))
                try:
                    idx = line_ngram.index(label)
                    res['ori_label_neutral'] = ori_label
                    res['neutral_examples'] = line
                    res['neutral_labels'] = [idx + i for i in range(len(label))]
                    res_list.append(res)
                    res = {
                        'ori_label': None,
                        'attributes_examples': (None, None),
                        'attributes_labels': None,
                        'ori_label_neutral': None,
                        'neutral_examples': None,
                        'neutral_labels': None,
                    }
                except:
                    pass
    
    return res_list


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    attributes_examples = [{} for _ in range(len(attributes_l))]
    attributes_labels = [{} for _ in range(len(attributes_l))]
    neutral_examples = []
    neutral_labels = []

    pbar = tqdm(data, desc=f'sentence')
    update = lambda *args: pbar.update()

    res = []
    n_proc = 8
    assert n_proc <= mp.cpu_count()
    pool = mp.Pool(n_proc)
    for orig_line in data:
        # print(orig_line)
        tmp = pool.apply_async(process, args=(orig_line, ), callback=update)
        res.append(tmp)
    pool.close()
    pool.join()
    new_res = []
    print("The number of results returned:", len(res))
    for r in res:
        r_resolved = r.get()
        new_res += r_resolved

    for d in new_res:
        if d['ori_label'] is not None:
            ori_label = d['ori_label']
            i = d['attributes_examples'][0]
            if ori_label in attributes_examples[i].keys():
                attributes_examples[i][ori_label].append(d['attributes_examples'][1])
                attributes_labels[i][ori_label].append(d['attributes_labels'])
            else:
                attributes_examples[i][ori_label] = [d['attributes_examples'][1]]
                attributes_labels[i][ori_label] = [d['attributes_labels']]

        elif d['ori_label_neutral'] is not None:
            neutral_examples.append(d['neutral_examples'])
            neutral_labels.append(d['neutral_labels'])
        else:
            pass

    attributes_examples_buffer = [[] for _ in range(len(attributes_l))]
    attributes_labels_buffer = [[] for _ in range(len(attributes_l))]
    for attributes in zip(*(sequential_l)):
        try:
            if args.ab_test_type == 'final':
                min_size = min([len(attributes_examples[i][a]) for i, a in enumerate(attributes)])
                # if min_size < 30:
                #     continue
                for i, a in enumerate(attributes):
                    attributes_examples_buffer[i] += attributes_examples[i][a][:min_size]
                    attributes_labels_buffer[i] += attributes_labels[i][a][:min_size]
            else:
                raise Exception()
        except:
            continue

    attributes_examples = attributes_examples_buffer
    attributes_labels = attributes_labels_buffer

    with open(args.output + '/count.txt', 'a') as wf:
        print('neutral:', len(neutral_examples), file=wf)
        for i, examples in enumerate(attributes_examples):
            print(f'attributes{i}:', len(examples), file=wf)
        print('neutral:', len(neutral_examples))
        for i, examples in enumerate(attributes_examples):
            print(f'attributes{i}:', len(examples))

    data = {'attributes_examples': attributes_examples,
            'attributes_labels': attributes_labels,
            'neutral_examples': neutral_examples,
            'neutral_labels': neutral_labels}

    torch.save(data, args.output + '/data.bin')
    print("Finish the data writing.")
