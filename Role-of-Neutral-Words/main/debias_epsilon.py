import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import logging
import time
import torch
import random
import numpy as np
import sys

from typing import List
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelWithLMHead,
    AdamW,
    # get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
timestr = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))


def inner_product(x, y):
    return torch.mean(torch.sum(y * x, 3))


def mean_square(x, y, idx):
    return torch.mean(torch.mean((y - x) ** 2, idx))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--config_name', type=str, default=None)
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--model_revision', type=str, default='main')

    parser.add_argument('--algorithm', type=str, default='DPCE')
    parser.add_argument('--bias', type=str, default='gender',
                        choices=["gender", "scm_gender", "religion", "race", "scm"])
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--loss_target', type=str, default='token')
    parser.add_argument('--debias_layer', type=str, default='all', choices=['all', 'first', 'last'])
    parser.add_argument('--dev_data_size', type=int, default=1000)
    parser.add_argument('--train_data_size', type=int, default=500)
    # parser.add_argument('--train_data_ratio', type=float, default=None)
    parser.add_argument('--weighted_loss', type=float, nargs=2, required=True)
    parser.add_argument('--KL_divergence', type=bool, default=False)
    parser.add_argument('--use_neutral', type=bool, default=True)

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--overwrite_output_dir', type=bool, default=False)
    parser.add_argument('--do_train', action='store_true', default=True)
    parser.add_argument('--do_eval', action='store_true', default=True)
    parser.add_argument('--local_rank', type=int, default=-1)

    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    # parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--num_train_epochs', type=int, required=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, required=True)    # 5e-5
    parser.add_argument('--lr_end', type=float, required=True)
    parser.add_argument('--power', type=int, default=5)

    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--random', action='store_true')
    parser.add_argument('--loss_neutral', action='store_true', help="Add loss_reg for neutral words or not.")
    args = parser.parse_args()
    print(f"model_name_or_path:{args.model_name_or_path}\n"  
          f"algorithm:{args.algorithm}  bias:{args.bias}\n"
          f"data_file:{args.data_file}\n"
          f"output_dir:{args.output_dir}\n"
        #   f"train_data_ratio:{args.train_data_ratio}\n"
          f"train_data_size:{args.train_data_size}  dev_data_size:{args.dev_data_size}\n"
          f"per_device_train_batch_size:{args.per_device_train_batch_size}  \
          per_device_eval_batch_size:{args.per_device_eval_batch_size}\n"
          f"weighted_loss: {str(args.weighted_loss[0])} {str(args.weighted_loss[1])}  learning_rate:{args.learning_rate}  lr_end:{args.lr_end}\n"
          f"logging_steps:{args.logging_steps}  random:{args.random}  loss_neutral:{args.loss_neutral}")
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def split_data(data, args):
    attributes_examples = data['attributes_examples']
    attributes_labels = data['attributes_labels']
    neutral_examples = data['neutral_examples']
    neutral_labels = data['neutral_labels']

    splited_data = {'train': {'example': {}, 'label': {}}, 'dev': {'example': {}, 'label': {}}}

    for i, (examples, labels) in enumerate(zip(attributes_examples, attributes_labels)):
        idx_l = list(range(len(examples)))
        random.shuffle(idx_l)
        examples = [examples[idx] for idx in idx_l]
        labels = [labels[idx] for idx in idx_l]
        splited_data['train']['example'][f'attribute{i}'] = examples[args.dev_data_size:]
        splited_data['train']['label'][f'attribute{i}'] = labels[args.dev_data_size:]
        splited_data['dev']['example'][f'attribute{i}'] = examples[:args.dev_data_size]
        splited_data['dev']['label'][f'attribute{i}'] = labels[:args.dev_data_size]

        print(f"Attribute subgroup-{i} train-size:{len(splited_data['train']['example'][f'attribute{i}'])} \
                dev-size:{len(splited_data['dev']['example'][f'attribute{i}'])}")
        
    idx_l = list(range(len(neutral_examples)))
    random.shuffle(idx_l)
    neutral_examples = [neutral_examples[idx] for idx in idx_l]
    splited_data['train']['example']['neutral'] = neutral_examples[args.dev_data_size:]
    splited_data['dev']['example']['neutral'] = neutral_examples[:args.dev_data_size]
    if neutral_labels is not None:
        neutral_labels = [neutral_labels[idx] for idx in idx_l]
        splited_data['train']['label']['neutral'] = neutral_labels[args.dev_data_size:]
        splited_data['dev']['label']['neutral'] = neutral_labels[:args.dev_data_size]
    print(f"Neutral train-size:{len(splited_data['train']['example']['neutral'])} \
        dev-size:{len(splited_data['dev']['example']['neutral'])}")

    return splited_data


def split_data_scm(data, args):
    splited_data = split_data(data, args)
    scm_data = {
        'train_warmth': {
            'example': {
                'attribute0': splited_data['train']['example']['attribute0'],
                'attribute1': splited_data['train']['example']['attribute1'],
                'neutral': splited_data['train']['example']['neutral']
            },
            'label': {
                'attribute0': splited_data['train']['label']['attribute0'],
                'attribute1': splited_data['train']['label']['attribute1'],
                'neutral': splited_data['train']['label']['neutral']
            },
        },
        'train_competence': {
            'example': {
                'attribute0': splited_data['train']['example']['attribute2'],
                'attribute1': splited_data['train']['example']['attribute3'],
                'neutral': splited_data['train']['example']['neutral']
            },
            'label': {
                'attribute0': splited_data['train']['label']['attribute2'],
                'attribute1': splited_data['train']['label']['attribute3'],
                'neutral': splited_data['train']['label']['neutral']
            },
        },
        'dev_warmth': {
            'example': {
                'attribute0': splited_data['dev']['example']['attribute0'],
                'attribute1': splited_data['dev']['example']['attribute1'],
                'neutral': splited_data['dev']['example']['neutral']
            },
            'label': {
                'attribute0': splited_data['dev']['label']['attribute0'],
                'attribute1': splited_data['dev']['label']['attribute1'],
                'neutral': splited_data['dev']['label']['neutral']
            },
        },
        'dev_competence': {
            'example': {
                'attribute0': splited_data['dev']['example']['attribute2'],
                'attribute1': splited_data['dev']['example']['attribute3'],
                'neutral': splited_data['dev']['example']['neutral']
            },
            'label': {
                'attribute0': splited_data['dev']['label']['attribute2'],
                'attribute1': splited_data['dev']['label']['attribute3'],
                'neutral': splited_data['dev']['label']['neutral']
            },
        },
    }

    return scm_data


class LineByLineTextDataset(Dataset):
    def __init__(self, examples: list, labels: list):
        self.examples = examples
        self.labels = labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.labels:
            return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.long)
        else:
            return torch.tensor(self.examples[i], dtype=torch.long)


def create_dataset(data, dataset):
    d = dict()
    for key in data['example'].keys():
        if key not in data['label']:
            d[key] = dataset(data['example'][key], None)
        else:
            d[key] = dataset(data['example'][key], data['label'][key])

    return d


def load_and_cache_examples(data):
    train_dataset = create_dataset(data['train'], LineByLineTextDataset)
    dev_dataset = create_dataset(data['dev'], LineByLineTextDataset)
    return {'train': train_dataset, 'dev': dev_dataset}


def load_and_cache_examples_scm(data):
    train_warmth_dataset = create_dataset(data['train_warmth'], LineByLineTextDataset)
    dev_warmth_dataset = create_dataset(data['dev_warmth'], LineByLineTextDataset)
    train_competence_dataset = create_dataset(data['train_competence'], LineByLineTextDataset)
    dev_competence_dataset = create_dataset(data['dev_competence'], LineByLineTextDataset)
    return {
        'warmth': {
            'train': train_warmth_dataset, 'dev': dev_warmth_dataset
        },
        'competence': {
            'train': train_competence_dataset, 'dev': dev_competence_dataset
        },
    }


def create_dataloader(args, datasets, tokenizer, train=False):
    def collate(batch: List[torch.Tensor]):
        if type(batch[0]) == tuple:
            examples, labels = list(zip(*batch))
            padded_examples = pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
            examples_attention_mask = torch.zeros_like(padded_examples, dtype=torch.int32)
            examples_attention_mask[torch.where(padded_examples != tokenizer.pad_token_id)] = 1
            padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
            labels_attention_mask = torch.zeros_like(padded_labels, dtype=torch.int32)
            labels_attention_mask[torch.where(padded_labels != 0)] = 1
            return padded_examples, padded_labels, examples_attention_mask, labels_attention_mask
        else:
            padded_examples = pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
            examples_attention_mask = torch.zeros_like(padded_examples, dtype=torch.int32)
            examples_attention_mask[torch.where(padded_examples != tokenizer.pad_token_id)] = 1
            return padded_examples, examples_attention_mask

    dataloaders = {}
    example_num = 0
    data_distribution = []

    min_size = min([len(value) for key, value in datasets.items() if key != 'neutral'])

    for key, dataset in datasets.items():
        example_num += len(dataset)
        if train:
            dataloaders[key] = iter(DataLoader(dataset, batch_size=args.train_batch_size, collate_fn=collate, shuffle=True))
            data_distribution += [key for _ in range(int(min_size / args.train_batch_size))]
        else:
            dataloaders[key] = iter(DataLoader(dataset, batch_size=args.eval_batch_size, collate_fn=collate , shuffle=False))
            data_distribution += [key for _ in range(int(min_size / args.eval_batch_size))]

    return dataloaders, example_num, data_distribution


class EXP_DEBIAS:
    def __init__(self, args):
        if args.config_name:
            self.config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir, revision=args.model_revision)
        elif args.model_name_or_path:
            self.config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, revision=args.model_revision)
        else:
            raise ValueError("You are instantiating a new config instance from scratch.")
        self.config.output_hidden_states = 'true'

        if args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError("You are instantiating a new tokenizer from scratch.")

        if args.block_size > 0:
            try:
                self.block_size = min(args.block_size, self.tokenizer.model_max_length)
            except:
                self.block_size = min(args.block_size, self.tokenizer.max_len)
        else:
            self.block_size = self.tokenizer.model_max_length

        self.model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
            cache_dir=args.cache_dir,
        )
        self.original_model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
            cache_dir=args.cache_dir,
        )

        # GPT-2 and GPT do not have pad.
        if 'pad_token' not in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.original_model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(args.device)
        self.original_model.to(args.device)

        self.data = torch.load(args.data_file)

        if args.bias == 'scm':
            self.splited_data = split_data_scm(self.data, args)
            self.datasets = load_and_cache_examples_scm(self.splited_data)
        else:
            self.splited_data = split_data(self.data, args)
            self.datasets = load_and_cache_examples(self.splited_data)

        self.num_train_epochs = args.num_train_epochs
        self.init_t_total(self.datasets, args)

        # Prepare optimizer and scheduler (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        # self.scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=self.t_total,
        # )
        self.scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=self.train_iter_num,
            lr_end=args.lr_end,
            power=args.power
        )
        self.criterion_ms = mean_square
        self.criterion_ip = inner_product

        self.alpha, self.beta = args.weighted_loss
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)

        self.best_loss = float('inf')
        self.best_step = 0
        self.global_step = 0
        self.epochs_trained = 0

        self.train_datasets = None
        self.dev_datasets = None
        self.train_distribution = None
        self.dev_distribution = None
        self.train_dataloaders = None
        self.dev_dataloaders = None

    def init_t_total(self, datasets, args):
        if args.bias != 'scm':
            train_datasets = datasets['train']
            _, _, train_distribution = create_dataloader(args, train_datasets, self.tokenizer, train=True)
            self.train_iter_num = self.num_train_epochs * len(train_distribution)
        else:
            warmth_train_datasets = datasets['warmth']['train']
            competence_train_datasets = datasets['competence']['train']
            _, _, warmth_train_distribution = create_dataloader(
                args, warmth_train_datasets, self.tokenizer, train=True)
            _, _, competence_train_distribution = create_dataloader(
                args, competence_train_datasets, self.tokenizer, train=True)
            self.train_iter_num = self.num_train_epochs * (
                    len(warmth_train_distribution) + len(competence_train_distribution))

    def get_hiddens_of_model(self, args, input, input_attention_mask):
        self.model.zero_grad()
        if args.model_type == 'roberta' and args.algorithm == 'DPCE':
                hiddens = self.model.roberta(input, input_attention_mask).hidden_states
        elif args.model_type == 'bert' and args.algorithm == 'DPCE':
                hiddens = self.model.bert(input, input_attention_mask).hidden_states
        else:
            raise AssertionError("Unknown model type!")
        return hiddens

    def attribute_vector_example(self, args, scm_dim_name=None):
        d = {'gender': 2, 'scm_gender': 2, 'religion': 3, 'scm': 2, 'warmth': 2, 'competence': 2}[args.bias]
        attributes_hiddens = {f'attribute{i}': [] for i in range(d)}

        dataloaders, _, distribution = create_dataloader(args, self.train_datasets, self.tokenizer, train=True)
        for key in distribution:
            if key != 'neutral':
                inputs, labels, inputs_attention_mask, _ = next(dataloaders[key])
                inputs = inputs.to(args.device)
                inputs_attention_mask = inputs_attention_mask.to(args.device)
                hiddens = self.get_hiddens_of_model(args, inputs, inputs_attention_mask)
                hiddens = torch.stack(hiddens, 2)
                if labels.size(1) > 1:
                    onehot = torch.eye(hiddens.size(1))
                    zeros = torch.zeros(1, onehot.size(0))
                    onehot = torch.cat((zeros, onehot), 0)
                    onehot = onehot[labels]
                    onehot = torch.sum(onehot, 1)
                    onehot = onehot.view(hiddens.size(0), -1, 1, 1)
                else:
                    onehot = torch.eye(hiddens.size(1))[labels].view(hiddens.size(0), -1, 1, 1)
                onehot = onehot.to(args.device)
                attributes_hiddens[key].append(torch.sum(hiddens * onehot, 1) / labels.size(1))
        if args.bias == 'scm':
            attribute_size = len(self.splited_data[f'train_{scm_dim_name}']['example'])
        else:
            attribute_size = len(self.splited_data['train']['example'])
        for i in range(attribute_size - 1):
            attributes_hiddens[f'attribute{i}'] = torch.mean(torch.cat(attributes_hiddens[f'attribute{i}'], 0),
                                                             0).detach().unsqueeze(0)

        return attributes_hiddens

    def forward(self, args, attributes_hiddens, dataloaders, key):
        """
        attributes_hiddens: {'attribute0'=Tensor:(1, 25, 1024), 'attribute0'=Tensor:(1, 25, 1024)}
        dataloaders: {'attribute0', 'attribute1', 'neutral'}
        key: 'attribute1'
        """
        inputs = next(dataloaders[key])
        if len(inputs) == 4:
            inputs, labels, inputs_attention_mask, labels_attention_mask = inputs
            labels = labels.to(args.device)
        else:
            inputs, inputs_attention_mask = inputs
            labels = None
        inputs, inputs_attention_mask = inputs.to(args.device), inputs_attention_mask.to(args.device)

        if args.model_type in ['bert', 'roberta']:
            all_layer_hiddens = self.model(inputs, inputs_attention_mask).hidden_states  # {Tensor:(16, 46, 768)} * 13
            all_layer_hiddens = torch.stack(all_layer_hiddens, 2)  # {Tensor:(16, 46, 13, 768)}
        else:
            raise AssertionError("Unknown model type!")

        # if 'neutral' != key:
        with torch.no_grad():
            all_layer_original_hiddens = self.original_model(inputs, inputs_attention_mask).hidden_states
            all_original_hiddens = torch.stack(all_layer_original_hiddens, 2)
            all_original_hiddens = all_original_hiddens.detach()

        if args.debias_layer == 'all':
            target_layer_hiddens = all_layer_hiddens
            target_original_hiddens = all_layer_hiddens
        else:
            raise AssertionError("Unknown debias layer!")

        if args.loss_target == 'token':
            if labels.size(1) > 1:
                onehot = torch.eye(target_layer_hiddens.size(1))
                zeros = torch.zeros(1, onehot.size(0))
                onehot = torch.cat((zeros, onehot), 0)
                onehot = onehot.to(args.device)
                onehot = onehot[labels]
                onehot = torch.sum(onehot, 1)
                onehot = onehot.view(target_layer_hiddens.size(0), -1, 1, 1)
            else:
                onehot = torch.eye(target_layer_hiddens.size(1))
                onehot = onehot.to(args.device)
                onehot = onehot[labels].view(target_layer_hiddens.size(0), -1, 1, 1)
            target_layer_hiddens = torch.sum(target_layer_hiddens * onehot, 1).unsqueeze(1) / labels.size(1)
            if 'neutral' != key:
                target_original_hiddens = torch.sum(target_original_hiddens * onehot, 1).unsqueeze(1) / labels.size(1)
            else:
                if args.algorithm == 'DPCE':
                    attributes_hiddens = {
                        key: value.expand(target_layer_hiddens.size(0), 1, value.size(1), value.size(2))
                        for key, value in attributes_hiddens.items()
                    }
                else:
                    raise AssertionError("Unknown algorithm type!")
        else:
            raise AssertionError("Unknown loss type!")

        if 'neutral' == key:
            loss = 0
            for attribute_hiddens in attributes_hiddens.values():
                tmp_loss = self.criterion_ip(target_layer_hiddens, attribute_hiddens)
                tmp_loss = tmp_loss ** 2
                tmp_loss *= self.alpha
                loss += tmp_loss
            
            if not args.loss_neutral:
                print(f"\n Loss_bias:{loss}")
            else:
                loss_neutral = self.criterion_ms(all_layer_hiddens, all_original_hiddens, 3)
                
                # -------------------------
                epsilon = 0.1 # 0.25
                if loss_neutral < epsilon:
                    loss_neutral = 0.0
                # -------------------------
                
                loss_neutral *= self.beta
                print(f"\n Loss_bias:{loss}     Loss neutral:{loss_neutral}")
                
                loss += loss_neutral
        else:
            loss = self.criterion_ms(all_layer_hiddens, all_original_hiddens, 3)
            loss *= self.beta
            print(f"\n Loss_reg:{loss} lr={self.scheduler.get_lr()[0]:<20}")

        return loss

    def evaluate(self, args, _attributes_hiddens):
        os.makedirs(args.output_dir, exist_ok=True)
        eval_loss = 0.0
        self.model.eval()
        for key in tqdm(self.dev_distribution):
            with torch.no_grad():
                loss = self.forward(args, _attributes_hiddens, self.dev_dataloaders, key)
                eval_loss += loss.item()

        self.dev_dataloaders, dev_example_num, self.dev_distribution = create_dataloader(
            args, self.dev_datasets, self.tokenizer, train=False)

        return eval_loss

    def save_best_model(self, args, attributes_hiddens):
        eval_loss = self.evaluate(args, attributes_hiddens)
        print("global_step = %s, evaluate loss = %s", self.global_step, eval_loss)

        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.best_step = self.global_step
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, timestr, 'best_model_ckpt')
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (self.model.module if hasattr(self.model, "module") else self.model)
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            print("Saving model checkpoint to %s", output_dir)

            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            print("Saving optimizer and scheduler states to %s", output_dir)
        print("best_step = %s, best loss = %s", self.best_step, self.best_loss)

    def train(self, args):
        self.model.zero_grad()
        self.original_model.zero_grad()
        self.original_model.eval()

        train_loss = 0.0
        train_iterator = trange(0, int(self.num_train_epochs), desc="Epoch")
        for train_epoch in train_iterator:
            # 防止stopIteration error
            self.train_datasets = self.datasets['train']
            self.dev_datasets = self.datasets['dev']

            self.train_dataloaders, _, self.train_distribution = create_dataloader(
                args, self.train_datasets, self.tokenizer, train=True
            )
            self.dev_dataloaders, _, self.dev_distribution = create_dataloader(
                args, self.dev_datasets, self.tokenizer, train=False
            )

            random.shuffle(self.train_distribution)
            epoch_iterator = tqdm(self.train_distribution, desc="Iteration")

            self.model.eval()
            with torch.no_grad():
                attributes_hiddens = self.attribute_vector_example(args)

            for step, key in enumerate(epoch_iterator):
                self.model.train()
                loss = self.forward(args, attributes_hiddens, self.train_dataloaders, key)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                train_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    self.original_model.zero_grad()
                    self.global_step += args.train_batch_size
                    # print(f"global steps: {self.global_step}")
                    # if args.logging_steps > 0 and self.global_step % (args.logging_steps * args.train_batch_size) == 0:
                    #     logger.info("global_step = %s, train loss = %s", self.global_step, train_loss)
                    #     train_loss = 0.0
                    #     self.save_best_model(args, attributes_hiddens)

            
            print("global_step = %s, train loss = %s", self.global_step, train_loss)
            train_loss = 0.0
            # self.save_best_model(args, attributes_hiddens)

            # --------------------------------------------------------
            if train_epoch >= 0 and (train_epoch + 1) % 1 == 0:
                eval_loss = self.evaluate(args, attributes_hiddens)
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.best_step = self.global_step

                print("global_step = %s, evaluate loss = %s", self.global_step, eval_loss)

                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, str(train_epoch), 'best_model_ckpt')
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                print("Saving model checkpoint to %s", output_dir)

                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                print("Saving optimizer and scheduler states to %s", output_dir)
            # --------------------------------------------------------

        # self.save_best_model(args, attributes_hiddens)
        print("best_step = %s, best loss = %s", self.best_step, self.best_loss)


class SCM_DEBIAS(EXP_DEBIAS):
    def __init__(self, args):
        super().__init__(args)

    def train(self, args):
        self.model.zero_grad()
        self.original_model.zero_grad()
        self.original_model.eval()

        train_loss = 0.0
        train_iterator = trange(0, int(self.num_train_epochs), desc="Epoch")
        for train_epoch in train_iterator:
            for scm_dim_name in ['warmth', 'competence']:
                self.train_datasets = self.datasets[scm_dim_name]['train']
                self.dev_datasets = self.datasets[scm_dim_name]['dev']

                self.train_dataloaders, _, self.train_distribution = create_dataloader(
                    args, self.train_datasets, self.tokenizer, train=True
                )
                self.dev_dataloaders, _, self.dev_distribution = create_dataloader(
                    args, self.dev_datasets, self.tokenizer, train=False
                )

                random.shuffle(self.train_distribution)
                epoch_iterator = tqdm(self.train_distribution, desc=f"Iteration ({scm_dim_name})")

                self.model.eval()
                with torch.no_grad():
                    attributes_hiddens = self.attribute_vector_example(args, scm_dim_name)

                for step, key in enumerate(epoch_iterator):
                    self.model.train()
                    loss = self.forward(args, attributes_hiddens, self.train_dataloaders, key)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    train_loss += loss.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                        self.optimizer.step()
                        self.scheduler.step()
                        self.model.zero_grad()
                        self.original_model.zero_grad()
                        self.global_step += args.train_batch_size
                        # print(f"global steps: {self.global_step}")
                        # if args.logging_steps > 0 and self.global_step % (args.logging_steps * args.train_batch_size) == 0:
            
            print("global_step = %s, train loss = %s", self.global_step, train_loss)
            train_loss = 0.0
            self.save_best_model(args, attributes_hiddens)

            # --------------------------------------------------------
            if train_epoch > 0 and (train_epoch + 1) % 1 == 0:
                eval_loss = self.evaluate(args, attributes_hiddens)
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.best_step = self.global_step

                print("global_step = %s, evaluate loss = %s", self.global_step, eval_loss)

                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, str(train_epoch), 'best_model_ckpt')
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                print("Saving model checkpoint to %s", output_dir)

                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                print("Saving optimizer and scheduler states to %s", output_dir)
            # --------------------------------------------------------

        # self.save_best_model(args, attributes_hiddens)
        print("best_step = %s, best loss = %s", self.best_step, self.best_loss)


def main():
    args = get_args()
    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        raise AssertionError("Local_rank is not -1!")

    set_seed(args)
    print("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)

    if args.bias == "scm":
        scm_debias = SCM_DEBIAS(args)
        scm_debias.train(args)
    else:
        exp_debias = EXP_DEBIAS(args)
        exp_debias.train(args)


if __name__ == "__main__":
    main()
    print("done!")
