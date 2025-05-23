import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CD2_Physio
from dataset_physio import get_dataloader
from utils import train, evaluate

import logging

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    filename='test_'+current_time+'.log',
    filemode='w',
)

import sys 
sys.argv += "--testmissingratio 0.1 --nsample 100".split()

parser = argparse.ArgumentParser(description="cd2")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:2', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="ph_80")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
logging.info(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

logging.info(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/physio_fold" + str(args.nfold) + "_" + current_time + "/"
logging.info('model folder:%s', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

torch.set_float32_matmul_precision("highest")

model = CD2_Physio(config, args.device).to(args.device)

model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
