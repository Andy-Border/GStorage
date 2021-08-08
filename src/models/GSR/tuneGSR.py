import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from utils import *

from models.GSR.trainGSR import train_GSR
from models.GSR.config import GSRConfig
import argparse
import numpy as np

rm_ratio_list = [0, 0.05, 0.1, 0.2, 0.4, 0.6]
rm_ratio_list = [0, 0.2, 0.4, 0.6]
add_ratio_list = [0, 1.0, 2.0]
add_ratio_list = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
zero_to_one_rough_list = [0, 0.25, 0.5, 0.75, 1.0]
zero_to_half_rough_list = [0, 0.25, 0.5]
small_list = [0, 0.25, 0.5]
fsim_weight_list = [0, 0.25, 0.5, 0.75, 1.0]
p_epoch_list = [100, 0, 5, 10, 20, 50, 100]
zero_to_one_fine_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# rm_cand_ratio = [0, 0.025, 0.05]
# add_cand_ratio = [0, 0.05, 0.1]
# filter_mode_list = [f'Mod_{i}_{j}' for i in rm_cand_ratio for j in add_cand_ratio]
c_conf_dict = {  # 50 x Trials
    'semb': 'dw',
    'fsim_weight': zero_to_one_rough_list,  # 5
    'intra_weight': zero_to_one_rough_list,  # 5
    'fsim_norm': True,
    'stochastic_trainer': False,
    'activation': ['Elu'],
    'p_batch_size': [128, 256, 512],
    'p_schedule_step': [0, 100, 250, 500, 1000]
}
fan_out_list = ['1_2', '3_5', '5_10', '10_20', '15_30', '20_40', '30_50']
EXP_DICT = {
    'RoughTune': {
        **c_conf_dict
        ,
        'data_spec_configs': {
            'fan_out': {
                'cora': '20_40',  # 5
                'citeseer': '10_20',  # 5
                'airport': '5_10',  # 2
                'blogcatalog': '15_30',
                'flickr': '15_30',
                'arxiv': '5_10',
            },
            'add_ratio': {
                'cora': zero_to_one_rough_list,  # 5
                'citeseer': zero_to_one_rough_list,  # 5
                'airport': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],  # 2
                'blogcatalog': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                'flickr': [0.0, 0.1, 0.2, 0.3],
                'arxiv': [0.0, 0.1, 0.2],
            },
            'rm_ratio': {
                'cora': [0.0],  # 2
                'citeseer': [0.0],
                'airport': [0.0, 0.25, 0.5],
                'blogcatalog': [0.0, 0.1, 0.2, 0.3],
                'flickr': [0.0, 0.1, 0.2, 0.3],
                'arxiv': [0.0, 0.05, 0.1],
            },
            'p_epochs': {
                'cora': [100, 10, 20, 30, 40, 50, 100],
                'citeseer': [0, 50, 100, 150, 200, 250, 300],
                'airport': [10, 20, 30, 40, 50, 100],
                'blogcatalog': [5, 1, 3, 5],
                'flickr': [5, 1, 3, 5],
                'arxiv': [3, 1, 2, 3]
            },
        },
    },
}
model_settings = {'model': 'GSR', 'model_config': GSRConfig, 'train_func': train_GSR}


@time_logger
def tune_mgsl():
    # * =============== Init Args =================
    exp_name = 'RoughTune'
    dataset = 'cora'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_times', type=int, default=5)
    parser.add_argument('-s', '--start_ind', type=int, default=0)
    parser.add_argument('-v', '--reverse_iter', action='store_true', help='reverse iter or not')
    parser.add_argument('-b', '--log_on', action='store_true', help='show log or not')

    parser.add_argument('-d', '--dataset', type=str, default=dataset)
    parser.add_argument('-e', '--exp_name', type=str, default=exp_name)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-t', '--train_percentage', type=int, default=0)
    args = parser.parse_args()
    args.__dict__.update(model_settings)
    if is_runing_on_local():
        args.gpu = -1
    # * =============== Fine Tune (grid search) =================

    tuner = Tuner(args, search_dict=EXP_DICT[args.exp_name])
    tuner.grid_search()
    tuner.summarize()


if __name__ == '__main__':
    tune_mgsl()

# tu -dcora -t1 -eFineTune1 -g0; tu -dcora -t3 -eFineTune1 -g0; tu -dcora -t5 -eFineTune1 -g0; tu -dcora -t10 -eFineTune1 -g0;

# python /home/zja/PyProject/MGSL/src/models/MGSL/sum_results.py


########################Aug 06 GNN backbone##########################

###cora
# tu -dcora -t0 -eRoughTune -g0 -oGCN; tu -dcora -t0 -eRoughTune -g0 -oGAT;  -dcora -t0 -eRoughTune -g0 -oGraphSage; tu -dcora -t0 -eRoughTune -g0 -oGCNII; tu -darxiv -t0 -eRoughTune -g0 -oGCNII;
#
# ### citeseer
# tu -dciteseer -t0 -eRoughTune -g1 -oGCN; tu -dciteseer -t0 -eRoughTune -g1 -oGAT; tu -dciteseer -t0 -eRoughTune -g1 -oGraphSage; tu -dciteseer -t0 -eRoughTune -g1 -oGCNII;
#
# ### airport
# tu -dairport -t0 -eRoughTune -g2 -oGCN; tu -dairport -t0 -eRoughTune -g2 -oGAT; tu -dairport -t0 -eRoughTune -g2 -oGraphSage; tu -dairport -t0 -eRoughTune -g2 -oGCNII;
#
# ### blogcatalog
# tu -dblogcatalog -t0 -eRoughTune -g3 -oGCN; tu -dblogcatalog -t0 -eRoughTune -g3 -oGAT; tu -dblogcatalog -t0 -eRoughTune -g3 -oGraphSage; tu -dblogcatalog -t0 -eRoughTune -g3 -oGCNII;
#
# ### flickr
# tu -dflickr -t0 -eRoughTune -g4 -oGCN; tu -dflickr -t0 -eRoughTune -g4 -oGAT; tu -dflickr -t0 -eRoughTune -g4 -oGraphSage; tu -dflickr -t0 -eRoughTune -g4 -oGCNII;
#
# ### arxiv
# tu -darxiv -t0 -eRoughTune -g5 -oGCN;
# tu -darxiv -t0 -eRoughTune -g6 -oGAT;
# tu -darxiv -t0 -eRoughTune -g7 -oGraphSage;
# #
