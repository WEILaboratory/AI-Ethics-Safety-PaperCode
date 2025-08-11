import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
)
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertForMaskedLM"
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased", 
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=None,
    help="RNG seed. Used for logging in experiment ID.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default='DPCE',
)


def load_model_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir, revision="main")
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer


if __name__ == "__main__":
    args = parser.parse_args()

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if 'gender' in args.model_name_or_path:
        bias_type = 'Gender'
    elif 'religion' in args.model_name_or_path:
        bias_type = 'Religion'
    elif 'scm' in args.model_name_or_path:
        bias_type = 'SCM'
    else:
        bias_type = 'original'

    # experiment_id = f"{args.model}_{bias_type}_{args.algorithm}"
    experiment_id = '_'.join(args.model_name_or_path.split('/')[1:-1])
    
    args.cache_dir = args.model_name_or_path
    model, tokenizer = load_model_tokenizer(args)
    model.to(device)
    model.eval()

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=_is_generative(args.model),
    )
    results = runner()

    os.makedirs(f"{args.persistent_dir}/evaluation/results/stereoset", exist_ok=True)
    f_name = f"{args.persistent_dir}/evaluation/results/stereoset/{experiment_id}.json"
    with open(f_name, "w") as f:
        json.dump(results, f, indent=2)
