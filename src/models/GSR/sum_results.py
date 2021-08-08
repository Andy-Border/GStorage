import os
import sys

sys.path.append((os.path.abspath(os.path.dirname(__file__)).split('src')[0] + 'src'))

from utils.proj_settings import RES_PATH
from utils.tune_utils import summarize_by_folder, summarize_by_tune_df, gen_tune_df
from models.GSR.tuneGSR import EXP_DICT
from models.GSR.config import GSRConfig

default_config_dict = {}


def summarize_results(datasets=['pubmed', 'cora', 'citeseer', 'arxiv', 'blogcatalog', 'flickr'],
                      exp_list=['RoughTune'],
                      train_percentage_list=[0, 1, 3, 5, 10],
                      models=[('GSR', GSRConfig)]):
    for dataset in datasets:
        for model, _ in models:
            try:
                summarize_by_folder(dataset, model)
            except:
                pass

    for dataset in datasets:
        for model, config in models:
            for exp_name in exp_list:
                for train_percentage in train_percentage_list:
                    trial_dict = {'dataset': dataset, 'exp_name': exp_name,
                                  'model': model, 'config': config, 'train_percentage': train_percentage}
                    tune_df = gen_tune_df(EXP_DICT[exp_name])
                    summarize_by_tune_df(tune_df, {**default_config_dict, **trial_dict})


if __name__ == "__main__":
    summarize_results()
# python /home/zja/PyProject/MGSL/src/models/GSR/sum_results.py
