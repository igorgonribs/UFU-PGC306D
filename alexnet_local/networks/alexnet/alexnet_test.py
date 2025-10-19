#!/usr/bin/env python3

"""alexnet_test.py

"""

import os
import sys
import glob
import yaml
import argparse

# Own modules (local sources)
sys.path.append("../../")
sys.path.append("./")
from lib.config import *
from lib.decor import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from alexnet_define import AlexNet

import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler

from torch import optim
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#===========================

if __name__ == "__main__":
  # -= init =-
  banner(f"Program -> {os.path.basename(__file__)}")
  args = parse_args()

  #sys.exit(0)
  #===============================================================================
















  sys.exit(0)
