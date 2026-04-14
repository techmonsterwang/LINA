# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Train a diffnext model."""

import argparse
import os
import sys
import subprocess

sys.path.append(os.getcwd())

from diffnext import engine
from diffnext.config import cfg
from diffnext.data import get_dataset_size
from diffnext.utils import logging


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Train a diffnext model")
    parser.add_argument("--cfg", default=None, help="config file")
    parser.add_argument("--exp-dir", default=None, help="experiment dir")
    parser.add_argument("--tensorboard", action="store_true", help="write metrics to tensorboard")
    parser.add_argument("--distributed", action="store_true", help="spawn distributed processes")
    parser.add_argument("--host", default="", help="hostfile for distributed training")
    parser.add_argument("--deepspeed", type=str, default="", help="deepspeed config file")
    parser.add_argument("--num-gpus", type=str, default="gpus per node")
    parser.add_argument("--world-size", type=str, default="total nodes")
    parser.add_argument("--rank", type=str, default="cycle rank use")
    parser.add_argument("--master-ip", type=str, default="job-xxxxx-master-0")
    return parser.parse_args()


def spawn_processes(args, coordinator):
    """Spawn distributed processes."""
    cmd = f"torchrun --nproc-per-node={args.num_gpus} --nnodes={args.world_size} --node-rank={args.rank} --master-addr={args.master_ip} "
    cmd += "{} --distributed".format(os.path.abspath(__file__))
    cmd += " --cfg {}".format(os.path.abspath(args.cfg))
    cmd += " --exp-dir {}".format(coordinator.exp_dir)
    cmd += " --tensorboard" if args.tensorboard else ""
    cmd += " --deepspeed {}".format(args.deepspeed) if args.deepspeed else ""
    return subprocess.call(cmd, shell=True), sys.exit()


def main(args):
    """Main entry point."""
    logging.info("Called with args:\n" + str(args))
    coordinator = engine.Coordinator(args.cfg, args.exp_dir)
    checkpoint, start_iter = coordinator.get_checkpoint()
    cfg.MODEL.WEIGHTS = checkpoint or cfg.MODEL.WEIGHTS
    logging.info("Using config:\n" + str(cfg))
    spawn_processes(args, coordinator) if cfg.NUM_GPUS > 1 else None
    engine.manual_seed(cfg.RNG_SEED, (cfg.GPU_ID, cfg.RNG_SEED))
    dataset_size = get_dataset_size(cfg.TRAIN.DATASET)
    logging.info("Dataset({}): {} examples for training.".format(cfg.TRAIN.DATASET, dataset_size))
    logging.info("Checkpoints will be saved to `{:s}`".format(coordinator.path_at("checkpoints")))
    engine.run_train(coordinator, start_iter, enable_tensorboard=args.tensorboard)


def main_distributed(args):
    """Main distributed entry point."""
    coordinator = engine.Coordinator(args.cfg, exp_dir=args.exp_dir)
    coordinator.deepspeed = args.deepspeed
    checkpoint, start_iter = coordinator.get_checkpoint()
    cfg.MODEL.WEIGHTS = checkpoint or cfg.MODEL.WEIGHTS
    engine.create_ddp_group(cfg)
    engine.manual_seed(cfg.RNG_SEED, (cfg.GPU_ID, cfg.RNG_SEED + engine.get_ddp_rank()))
    dataset_size = get_dataset_size(cfg.TRAIN.DATASET)
    logging.info("Dataset({}): {} examples for training.".format(cfg.TRAIN.DATASET, dataset_size))
    logging.info("Checkpoints will be saved to `{:s}`".format(coordinator.path_at("checkpoints")))
    engine.run_train(coordinator, start_iter, enable_tensorboard=args.tensorboard)


if __name__ == "__main__":
    args = parse_args()
    main_distributed(args) if args.distributed else main(args)
