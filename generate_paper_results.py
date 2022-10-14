import pandas as pd
from matplotlib import pyplot as plt
import os,sys
import matplotlib
import numpy as np
from analysis_utils import (load_log, plot_train_val_losses,
                            print_accs, print_best_wd_accs, plot_adj_sweep, print_best_adj_accs,
                            process_df, process_df_waterbird9)

additions = ['CL']

def get_dirpath(dataset, opt_type, wd, lr, adj=0):
    folder = f'{dataset}_{opt_type}_wd-{wd}_lr-{lr}'
    if opt_type == 'DRO':
        folder += f'_adj-{adj}'
    if opt_type == 'CL':
        folder += f'_adj-{adj}'
    return folder

cl_epoch = {'celebA': 19, 'waterbirds': 99, 'multiNLI': 7}

params = {}

params['celebA'] = {
    'n_groups': 4,
    'wd': [0.0001, 0.01, 0.1],
    'lr': {
        0.0001: 1e-4,
        0.01: 1e-4,
        0.1: 1e-5
    },
    'bs': 128,
    'n_epochs': 50,
    'adjusted_wd': 0.1,
    'adj_list': [0],
    'n_train': [71629, 66874, 22880, 1387],
    'n_val': [8535, 8276, 2874, 182],
    'n_test': [9767, 7535, 2480, 180],
    'process_df': process_df,
    'opt_types': ['ERM', 'DRO'] + additions
}

params['waterbirds'] = {
    'n_groups': 4,
    'wd':[0.0001, 0.1, 1],
    'lr': {
        0.0001: 1e-3,
        0.1: 1e-4,
        1: 1e-5
    },
    'bs': 128,
    'n_epochs': 300,
    'adjusted_wd': 1,
    'adj_list': [0],
    'n_train': [3498, 184, 56, 1057],
    'n_val': [467, 466, 133, 133],
    'n_test': [2255, 2255, 642, 642],
    'process_df': process_df_waterbird9,
    'opt_types': ['ERM', 'DRO'] + additions
}

# This refers to multiNLI trained to 20 epochs
params['multiNLI'] = {
    'n_groups': 6,
    'wd': [0],
    'lr': {
        0: 2e-5
    },
    'bs': 32,
    'n_epochs': 20,
    'adjusted_wd': 0,
    'adj_list': [0],
    'n_train': [57498, 11158, 67376, 1521, 66630, 1992],
    'n_val': [22814, 4634, 26949, 613, 26655, 797],
    'n_test': [34597, 6655, 40496, 886, 39930, 1148],
    'process_df': process_df,
    'opt_types': ['ERM', 'DRO'] + additions
}

# This is multiNLI trained to 3 epochs.
# Early stopping on multiNLI trained to 20 epochs is different from
# early stopping on multiNLI trained 3 epochs, since LR decay schedule differs.
params['multiNLI_3'] = params['multiNLI'].copy()
params['multiNLI_3']['n_epochs'] = 3
params['multiNLI_3']['opt_types'] = ['ERM', 'DRO'] + additions

dfs = {}

for dataset in params.keys():
    dfs[dataset] = {}
    loss_metrics = []
    acc_metrics = []
    for group_idx in range(params[dataset]['n_groups']):
        loss_metrics.append(f'avg_loss_group:{group_idx}')
        acc_metrics.append(f'avg_acc_group:{group_idx}')

    for wd in params[dataset]['wd']:
        for opt_type in ['ERM', 'DRO' ] + additions:
            if opt_type not in dfs[dataset]:
                dfs[dataset][opt_type] = {}
            if opt_type=='DRO':
                adj_list = params[dataset]['adj_list']
            elif opt_type=='CL':
                adj_list = params[dataset]['adj_list']
            else:
                adj_list = [0,]

            for adj in adj_list:
                if (wd != params[dataset]['adjusted_wd']) and (adj != 0): continue
                if adj not in dfs[dataset][opt_type]:
                    dfs[dataset][opt_type][adj] = {}

                train_df, val_df, test_df = load_log(
                        get_dirpath(
                            dataset,
                            opt_type,
                            lr=params[dataset]['lr'][wd],
                            wd=wd,
                            adj=adj))
                if test_df is None:
                    dfs[dataset][opt_type][adj][wd] = None
                    continue
                params[dataset]['process_df'](train_df, val_df, test_df, params[dataset])
                dfs[dataset][opt_type][adj][wd] = {'train': train_df, 'val': val_df, 'test': test_df}

#################################################################
#                            TABLE 1                            #
#################################################################
print('##### Table 1 #####')

# With early stopping
print('### Early stopping')
adj = 0
for dataset, wd, epoch_to_eval in [
    ('celebA', 0.0001, 0),
    ('waterbirds', 0.0001, 0),
    ('multiNLI_3', 0, 2),
    ]:
    for opt_type in ['ERM', 'DRO'] + additions:
        print(f'## {dataset} {opt_type} adj={adj} wd={wd}')

        df = dfs[dataset][opt_type][adj][wd]
        print_accs(
            df,
            params[dataset],
            epoch_to_eval=epoch_to_eval,
            print_avg=True,
            splits=['train', 'val', 'test'],
            early_stop=False
        )
        print()

# No regularization and early stopping
print('### Standard regularization')
dataset_wds = [
    ('celebA', 0.0001),
    ('waterbirds', 0.0001),
    ('multiNLI', 0),
]
adj = 0
for dataset, wd in dataset_wds:
    for opt_type in ['ERM', 'DRO'] + additions:
        print(f'## {dataset} {opt_type} adj={adj} wd={wd}')
        max_epoch = dfs[dataset][opt_type][adj][wd]['test']['epoch'].max()
        if opt_type == "CL":
            max_epoch = cl_epoch[dataset]
        df = dfs[dataset][opt_type][adj][wd]
        print_accs(
            df,
            params[dataset],
            epoch_to_eval=max_epoch,
            print_avg=True,
            splits=['train', 'test'],
            early_stop=False
        )
        print()

# With strong regularization
print('### Strong L2 regularization')
dataset_wds = [
    ('celebA', 0.1),
    ('waterbirds', 1.0),
]
adj = 0
for dataset, wd in dataset_wds:
    for opt_type in ['ERM', 'DRO'] + additions:
        print(f'## {dataset} {opt_type} adj={adj} wd={wd}')
        max_epoch = dfs[dataset][opt_type][adj][wd]['test']['epoch'].max()
        if opt_type == "CL":
            max_epoch = cl_epoch[dataset]
        df = dfs[dataset][opt_type][adj][wd]
        print_accs(
            df,
            params[dataset],
            epoch_to_eval=max_epoch,
            print_avg=True,
            splits=['train', 'val', 'test'],
            early_stop=False
        )
        print()


##################################################################
#                            FIGURE 2                            #
##################################################################
import warnings
warnings.filterwarnings('ignore')


ns = 3
def plot_figure2(weight_decays, options, params, dfs, num_epochs, place):
    groups = ['Dark hair, female', 'Dark hair, male', 'Blond, female', 'Blond, male']
    plt.rcParams.update({'font.size': 20, 'lines.linewidth':4})
    fig, ax = plt.subplots(2, ns,
                           figsize=(ns * 10,2.5),
                           sharey=True, sharex=True)
    acc=True
    for i_opt_type,opt_type in enumerate(options): #['CL']): # 'ERM','CL','DRO']): #, 'CL']):
        if acc:
            plotted_col='avg_acc'
        else:
            plotted_col='avg_loss'
        for i_wd, wd in enumerate(weight_decays):
            legend = []
            for group_idx in range(params['n_groups']):
                df = dfs[opt_type][0][wd]
                legend.append(groups[group_idx])
                legend.append('_no_legend_')
                if df is None:
                    print("ERROR")
                    print(opt_type, wd)
                    continue
                plot_train_val_losses(ax[i_wd, i_opt_type], df['train'], df['val'],
                                      f'{plotted_col}_group:{group_idx}', f'C{group_idx}',
                                      title=f'{opt_type}, wd={wd}')
            ax[i_wd, i_opt_type].set_xlabel('Training Time')
            if i_opt_type==0 and i_wd==0:
                ax[i_wd, i_opt_type].set_ylabel('Accuracy', labelpad=-10)
            else:
                ax[i_wd,i_opt_type].set_ylabel(None, labelpad=-10)
                ax[i_wd,i_opt_type].set_yticks([])
            if wd==0.0001:
                title = f'{opt_type}\nStandard Regularization'
            else:
                title = f'{opt_type}\nStrong $\ell_2$ Regularization'

            ax[i_wd,i_opt_type].set_title(title)
            ax[i_wd,i_opt_type].grid(b=None)
            ax[i_wd,i_opt_type].set_xticks([])

    ax[i_wd,i_opt_type].set_ylim((0,1.02))
    ax[i_wd,i_opt_type].set_yticks([0,1])
    print(np.max([df['train']['batch'].values]))
    # ax[ns*i_wd+i_opt_type].set_xlim([0,np.max([df['train']['batch'].values])])
    # ax[ns*i_wd+i_opt_type].set_xlim([0, num_epochs])
    ax[i_wd,i_opt_type].set_xticklabels([])
    ax[i_wd,i_opt_type].legend(legend, loc='lower right', ncol=2 * ns, bbox_to_anchor=(0.3,-0.45))
    fig.tight_layout()
    for i in range(2):
        for j in range(ns):
            # ax[i,j].set_position([i*0.25, 0, 0.24, .9], which='both')
            continue
    plt.savefig(place, bbox_inches='tight')

plot_figure2([0.0001,0.1], ["CL"], params['celebA'], dfs['celebA'], 20, "figure2.pdf")
plot_figure2([0.0001,0.1], ["ERM", "DRO"], params['celebA'], dfs['celebA'], 50, "figure22.pdf")

def plot_figure2(weight_decays, params, dfs):
    groups = ['Dark hair, female', 'Dark hair, male', 'Blond, female', 'Blond, male']
    plt.rcParams.update({'font.size': 20, 'lines.linewidth':4})
    fig, ax = plt.subplots(1,4,
                           figsize=(20,2.5),
                           sharey=True, sharex=True)
    acc=True
    for i_opt_type,opt_type in enumerate(['CL','DRO']):
        if acc:
            plotted_col='avg_acc'
        else:
            plotted_col='avg_loss'
        for i_wd, wd in enumerate(weight_decays):
            legend = []
            for group_idx in range(params['n_groups']):
                df = dfs[opt_type][0][wd]
                if df is None:
                    continue
                if opt_type == "CL":
                    title = "GDRO-R"
                else:
                    title = "GDRO"
                plot_train_val_losses(ax[2*i_wd+i_opt_type], df['train'], df['val'],
                                      f'{plotted_col}_group:{group_idx}', f'C{group_idx}',
                                      title=f'{title}, wd={wd}')
                legend.append(groups[group_idx])
                legend.append('_no_legend_')
            ax[i_wd*2+i_opt_type].set_xlabel('Training Time')
            if i_opt_type==0 and i_wd==0:
                ax[2*i_wd+i_opt_type].set_ylabel('Accuracy', labelpad=-10)
            else:
                ax[2*i_wd+i_opt_type].set_ylabel(None, labelpad=-10)
                ax[2*i_wd+i_opt_type].set_yticks([])
            if wd==0.0001:
                title = f'{title}\nStandard Regularization'
            else:
                title = f'{title}\nStrong $\ell_2$ Regularization'

            ax[2*i_wd+i_opt_type].set_title(title)
            ax[2*i_wd+i_opt_type].grid(b=None)
            ax[2*i_wd+i_opt_type].set_xticks([])
            print(df["train"].shape[0])
            print(df["train"])
            print(np.max([df["train"]["batch"].values]))

    #ax[2*i_wd+i_opt_type].set_ylim((0,1.02))
    ax[2*i_wd+i_opt_type].set_ylim((0.6,1.03))
    ax[2*i_wd+i_opt_type].set_yscale('log')
    ax[2*i_wd+i_opt_type].set_yticks([0.6,1])

    ax[2*i_wd+i_opt_type].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[2*i_wd+i_opt_type].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax[2*i_wd+i_opt_type].set_xlim([0,np.max([df['train']['batch'].values])])
    #ax[2*i_wd+i_opt_type].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax[2*i_wd+i_opt_type].set_xlim([0,50]) # np.max([df['train']['batch'].values])])
    ax[2*i_wd+i_opt_type].set_xticklabels([])
    ax[2*i_wd+i_opt_type].legend(legend, loc='lower right', ncol=4, bbox_to_anchor=(0.3,-0.45))
    fig.tight_layout()
    for i in range(4):
        ax[i].set_position([i*0.25, 0, 0.24, .9], which='both')
    plt.savefig('figure2.pdf', bbox_inches='tight')
plot_figure2([0.0001,0.1], params['celebA'], dfs['celebA'])
