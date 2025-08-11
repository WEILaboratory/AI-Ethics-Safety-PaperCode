import argparse
import os
from utils import setup_seed, construct_log, get_random_dir_name
from fls.flbase import MODEL
from fls.effl import EFFL

import time
import json
from tensorboardX import SummaryWriter
import torch
import shutil
from models import RegressionModel
import math

parser = argparse.ArgumentParser()

# environment configuration
parser.add_argument(
    "-dev", "--device", type=str, default="cuda", choices=["cpu", "cuda"]
)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("-did", "--device_id", type=str, default="1")
# Model training configuration ---- public
parser.add_argument(
    "--norm", type=str, default="loss+", choices=["l2", "loss", "loss+", "none"]
)
parser.add_argument("--step_size", type=float, default=0.03)

parser.add_argument("--batch_size", type=list, default=[100, 100])
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--drop_last", type=bool, default=False)
parser.add_argument(
    "--sampler",
    type=str,
    default="None",
)
# ******************Important*******************************************************#
parser.add_argument("--method", type=str, default="EFFL")

parser.add_argument(
    "--max_epoch_stage",
    nargs="+",
    type=int,
    default=[750, 750, 500],
)
parser.add_argument(
    "--disparity_type",
    type=str,
    default="TPSD",
)

parser.add_argument(
    "--attack_type",
    type=str,
    default="None",
    help="Random|Enlarge|Zero|None",
)

parser.add_argument("--attack_ratio", type=float, default=0.4)
# **********************************************************************************#
parser.add_argument("--eval_epoch", type=int, default=20)
parser.add_argument("--global_epoch", type=int, default=0)
parser.add_argument("--n_hiddens", type=int, default=20)
# Fairness budget
parser.add_argument("--eps_g", type=float, default=0.05, help="group fairness")
parser.add_argument(
    "--eps_vl", type=float, default=0.02, help="egalitarian fairness: loss variance"
)
parser.add_argument(
    "--eps_vg", type=float, default=0.02, help="egalitarian fairness: fairness variance"
)

# Dataset Settting
parser.add_argument("--sensitive_attr", type=str, default="race")
parser.add_argument(
    "--dataset", type=str, default="synthetic", help="[synthetic,adult, eicu]"
)
parser.add_argument("--num_workers", type=int, default=0)
# Paths
parser.add_argument(
    "--target_dir_name",
    type=str,
    default="test_out",
)

parser.add_argument("--log_name", type=str, default="log")
parser.add_argument("--data_dir", type=str, default="data")
# For FCFL
parser.add_argument("--eps_delta_l", type=float, default=1e-2)
parser.add_argument("--eps_delta_g", type=float, default=1e-2)
parser.add_argument("--factor_delta", type=float, default=1e-1)
parser.add_argument("--lr_delta", type=float, default=0.1)
parser.add_argument("--delta_l", type=float, default=0.5)
parser.add_argument("--delta_g", type=float, default=0.5)
# For qFed
parser.add_argument(
    "--q", help="reweighting factor", type=float, default="6.0"
)  # 0.0: no weighting, the same as fedavg

#  For FedReg
parser.add_argument(
    "--weight_fair", type=float, default=1.0, help="weight for disparity"
)
#   For FedMDFG
parser.add_argument("--s", help="line search parameter of FedMDFG", type=int, default=1)
parser.add_argument("--theta", type=float, default=1)
parser.add_argument("--force_active", type=bool, default=True)

# For Ditto
parser.add_argument("--local_epochs", type=int, default=5)
parser.add_argument("--lam", type=float, default=1)

# For FairFed
parser.add_argument("--beta", type=float, default=1)

args = parser.parse_args()
setup_seed(seed=args.seed)

args.target_dir_name = "results/" + args.target_dir_name
if os.path.exists(args.target_dir_name):
    shutil.rmtree(args.target_dir_name)

args.eps = [
    args.eps_g,
    args.eps_vl,
    args.eps_vg,
]

args.train_dir = os.path.join(args.data_dir, args.dataset, "train")
args.test_dir = os.path.join(args.data_dir, args.dataset, "test")
args.log_dir = os.path.join(args.target_dir_name)

# For FCFL
args.eps_delta = [
    args.eps_delta_l,
    args.eps_delta_g,
]

os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

if args.device == "cuda" and not torch.cuda.is_available():
    print("\ncuda is not avaiable.\n")
    args.device = "cpu"

if "synthetic" in args.dataset:
    args.n_feats = 2
if "adult" in args.dataset:
    args.n_feats = 94
    # args.model = RegressionModel(args.n_feats, 0)
if "eicu" in args.dataset:
    args.n_feats = 72
    # args.model = RegressionModel(args.n_feats, 0)


if __name__ == "__main__":
    os.makedirs(args.log_dir, exist_ok=True)
    logger = construct_log(args)

    model = EFFL(args, logger)
    model.train()
