#!/usr/bin/env python
import torch
from options import Options
from problem import Problem
from trainer import Trainer

args    = Options().parse()
trainer = Trainer(args)
trainer.train()
