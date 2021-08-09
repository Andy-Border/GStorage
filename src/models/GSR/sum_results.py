import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))

from utils import *
from models.GSR import *
from models.GSR.tuneGSR import EXP_DICT

DATASETS = ['cora', 'citeseer', 'blogcatalog', 'airport', 'flickr', 'arxiv']
EXP_LIST = ['RoughTuneGCN', 'RoughTuneGAT', 'RoughTuneGraphSage']
TRAIN_RATIOS = [0]

MODELS = [('GSR', GSRConfig, train_GSR)]


def summarize_results(datasets=DATASETS, exp_list=EXP_LIST,
                      train_percentage_list=TRAIN_RATIOS, models=MODELS):
    # ! Sumarize by datasets
    for dataset in datasets:
        for model, _, _ in models:
            summarize_by_folder(dataset, model)

    # ! Summarize by experiments
    for model, config, train_func in models:
        for dataset in datasets:
            for exp_name in exp_list:
                for train_percentage in train_percentage_list:
                    tuner = tuner_from_argparse_args(model, config, train_func, EXP_DICT,
                                                     dataset=dataset, train_percentage=train_percentage,
                                                     exp_name=exp_name)
                    tuner.summarize()


if __name__ == "__main__":
    summarize_results()
# pyt src/utils/sum_results.py
