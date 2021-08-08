import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))

from utils import *


@time_logger
def train_GSR(args):
    # ! Init Environment
    exp_init(args.seed if hasattr(args, 'seed') else 0, args.gpu)

    # ! Import packages
    # Note that the assignment of GPU-ID must be specified before torch/dgl is imported.
    import torch as th
    import dgl
    from utils.data_utils import preprocess_data
    from models.GSR import GSR_pretrain, GSR_finetune, para_copy, GSRConfig, \
        get_pretrain_loader, get_structural_feature, \
        MemoryMoCo, moment_update, NCESoftmaxLoss, FullBatchTrainer

    # ! Config
    cf = GSRConfig(args)
    cf.device = th.device("cuda:0" if args.gpu >= 0 else "cpu")

    # ! Load Graph
    g, features, cf.n_feat, cf.n_class, labels, train_x, val_x, test_x = \
        preprocess_data(cf.dataset, cf.train_percentage)
    feat = {'F': features, 'S': get_structural_feature(g, cf)}
    cf.feat_dim = {v: feat.shape[1] for v, feat in feat.items()}
    supervision = SimpleObject({'train_x': train_x, 'val_x': val_x, 'test_x': test_x, 'labels': labels})

    # ! Train Init
    print(f'{cf}\nStart training..')
    p_model = GSR_pretrain(g, cf).to(cf.device)
    print(p_model)

    # ! Train Phase 1: Pretrain
    if cf.p_epochs > 0:
        # os.remove(cf.pretrain_model_ckpt)  # Debug Only
        if os.path.exists(cf.pretrain_model_ckpt):
            p_model.load_state_dict(th.load(cf.pretrain_model_ckpt, map_location=cf.device))
            print(f'Pretrain embedding loaded from {cf.pretrain_model_ckpt}')
        else:
            print(f'>>>> PHASE 1 - Pretraining and Refining Graph Structure <<<<<')
            views = ['F', 'S']
            optimizer = th.optim.Adam(
                p_model.parameters(), lr=cf.prt_lr, weight_decay=cf.weight_decay)
            # Construct virtual relation triples
            p_model_ema = GSR_pretrain(g, cf).to(cf.device)
            moment_update(p_model, p_model_ema, 0)  # Copy
            moco_memories = {v: MemoryMoCo(cf.n_hidden, cf.nce_k,  # Single-view contrast
                                           cf.nce_t, device=cf.device).to(cf.device)
                             for v in views}
            criterion = NCESoftmaxLoss(cf.device)
            pretrain_loader = get_pretrain_loader(g.cpu(), cf)

            for epoch_id in range(cf.p_epochs):
                for step, (input_nodes, edge_subgraph, blocks) in enumerate(pretrain_loader):
                    t0 = time()
                    blocks = [b.to(cf.device) for b in blocks]
                    edge_subgraph = edge_subgraph.to(cf.device)
                    input_feature = {v: feat[v][input_nodes].to(cf.device) for v in views}

                    # ===================Moco forward=====================
                    p_model.train()

                    q_emb = p_model(edge_subgraph, blocks, input_feature, mode='q')
                    std_dict = {v: round(q_emb[v].std(dim=0).mean().item(), 4) for v in ['F', 'S']}
                    print(f"Std: {std_dict}")

                    if std_dict['F'] == 0 or std_dict['S'] == 0:
                        print(f'\n\n????!!!! Same Embedding Epoch={epoch_id}Step={step}\n\n')
                        # q_emb = p_model(edge_subgraph, blocks, input_feature, mode='q')

                    with th.no_grad():
                        k_emb = p_model_ema(edge_subgraph, blocks, input_feature, mode='k')
                    intra_out, inter_out = [], []

                    for tgt_view, memory in moco_memories.items():
                        for src_view in views:
                            if src_view == tgt_view:
                                intra_out.append(memory(
                                    q_emb[f'{tgt_view}'], k_emb[f'{tgt_view}']))
                            else:
                                inter_out.append(memory(
                                    q_emb[f'{src_view}->{tgt_view}'], k_emb[f'{tgt_view}']))

                    # ===================backward=====================
                    # ! Self-Supervised Learning
                    intra_loss = th.stack([criterion(out_) for out_ in intra_out]).mean()
                    inter_loss = th.stack([criterion(out_) for out_ in inter_out]).mean()
                    # ! Loss Fusion
                    loss_tensor = th.stack([intra_loss, inter_loss])
                    intra_w = float(cf.intra_weight)
                    loss_weights = th.tensor([intra_w, 1 - intra_w], device=cf.device)
                    loss = th.dot(loss_weights, loss_tensor)
                    # ! Semi-Supervised Learning
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    moment_update(p_model, p_model_ema, cf.momentum_factor)
                    print_log({'Epoch': epoch_id, 'Batch': step, 'Time': time() - t0,
                               'intra_loss': intra_loss.item(), 'inter_loss': inter_loss.item(),
                               'overall_loss': loss.item()})
                epochs_to_save = P_EPOCHS_SAVE_LIST + ([1, 2, 3, 4] if args.dataset == 'arxiv' else [])
                if epoch_id + 1 in epochs_to_save:
                    # Convert from p_epochs to current p_epoch checkpoint
                    ckpt_name = cf.pretrain_model_ckpt.replace(f'_pi{cf.p_epochs}', f'_pi{epoch_id + 1}')
                    th.save(p_model.state_dict(), ckpt_name)
                    print(f'Model checkpoint {ckpt_name} saved.')

            th.save(p_model.state_dict(), cf.pretrain_model_ckpt)

    # ! Train Phase 2: Graph Structure Refine
    print(f'>>>> PHASE 2 - Graph Structure Refine <<<<< ')

    if cf.p_epochs <= 0 or cf.add_ratio + cf.rm_ratio == 0:
        print('Use original graph!')
        g_new = g
    else:
        if os.path.exists(cf.refined_graph_file):
            print(f'Refined graph loaded from {cf.refined_graph_file}')
            g_new = dgl.load_graphs(cf.refined_graph_file)[0][0]
        else:
            g_new = p_model.refine_graph(g, feat)
            dgl.save_graphs(cf.refined_graph_file, [g_new])

    # ! Train Phase 3:  Node Classification
    f_model = GSR_finetune(cf).to(cf.device)
    print(f_model)
    # Copy parameters
    if cf.p_epochs > 0:
        para_copy(f_model, p_model.encoder.F, paras_to_copy=['conv1.weight', 'conv1.bias'])
    optimizer = th.optim.Adam(f_model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
    stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop else None
    del g, feat, p_model
    th.cuda.empty_cache()

    print(f'>>>> PHASE 3 - Node Classification <<<<< ')
    trainer_func = FullBatchTrainer
    trainer = trainer_func(model=f_model, g=g_new, features=features, sup=supervision, cf=cf,
                           stopper=stopper, optimizer=optimizer, loss_func=th.nn.CrossEntropyLoss())
    trainer.run()
    trainer.eval_and_save()

    return cf


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training settings")
    dataset = 'pubmed'
    dataset = 'citeseer'
    dataset = 'arxiv'
    dataset = 'flickr'
    dataset = 'blogcatalog'
    dataset = 'cora'
    # ! Exp Settings
    parser.add_argument("-g", "--gpu", default=1, type=int, help="GPU id to use.")
    parser.add_argument("-d", "--dataset", type=str, default=dataset)
    parser.add_argument("-t", "--train_percentage", default=-1, type=int)
    parser.add_argument("-e", "--early_stop", default=100, type=int)
    parser.add_argument('-l', '--load_default_config', action='store_true',
                        help='Whether load default config or use parsed config')
    parser.add_argument("--seed", default=0)

    # ! Model Settings
    # Note that model settings will be overwritten by default setting with "-l" option
    parser.add_argument("--epochs", default=1000, type=float)
    parser.add_argument("--intra_weight", default=0.5, type=float)
    parser.add_argument("--fsim_weight", default=0.25, type=float)
    parser.add_argument("--fan_out", default='20_40', type=str)
    parser.add_argument("--add_ratio", default=0.5, type=float)
    parser.add_argument("--rm_ratio", default=0, type=float)
    parser.add_argument("--p_epochs", default=50, type=int)
    parser.add_argument("--p_batch_size", default=512, type=int)
    parser.add_argument("--prt_lr", default=0.005, type=float)
    parser.add_argument("--activation", default='Relu', type=str)
    args = parser.parse_args()

    if is_runing_on_local():
        args.gpu = -1
        args.dataset = args.dataset if args.dataset != 'arxiv' else 'cora'
    # args.load_default_config = True
    if args.load_default_config:
        print('Please note that, the model settings are overwritten by default setting with "-l" option')
        for var_name, var_val_dict in DEFAULT_SETTING.items():
            args.__dict__[var_name] = var_val_dict[args.dataset]
    # ! Train
    train_GSR(args)

# python /home/zja/PyProject/MGSL/src/models/GSR/trainGSR.py -darxiv
