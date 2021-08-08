import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]
os.chdir(root_path)
sys.path.append(root_path + 'src')

from .early_stopper import EarlyStopping
from .util_funcs import exp_init, time_logger, print_log, is_runing_on_local
from .proj_settings import P_EPOCHS_SAVE_LIST, DEFAULT_SETTING
from .conf_utils import SimpleObject
import argparse
from time import time
