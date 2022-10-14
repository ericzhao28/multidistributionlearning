import os
from subprocess import call


### Shared parameter settings
a = {}
a['bs'] = 128
a['alpha'] = 0.01
a['gamma'] = 0.1
a['adj'] = 0
a['use_normalized_loss'] = False
a['augment_data'] = False
a['scheduler'] = False
a['frac'] = 1.0
a['n_epochs'] = 400
a['save_step'] = 10000
a['log_every'] = 50
a['model'] = 'resnet50'
a['dataset_cmd'] = "-s confounder -d CUB -t waterbird_complete95 -ea waterbird_complete95 forest2water2"
a['priority'] = None
a['gpu_type'] = None

def main():

    params = {}

    params['celebA'] = {
        'n_groups': 4,
        # 'wd': [0.0001, 0.01, 0.1],
        'wd': [0.0001, 0.1],
        'lr': {
            0.0001: 1e-4,
            0.01: 1e-4,
            0.1: 1e-5
        },
        'bs': 128,
        'n_epochs': 50,
        'adjusted_wd': 0.1,
        'adj_list': [0, 1, 2, 3, 4, 5],
        'command': '-s confounder -d CelebA -t Blond_Hair -c Male --model resnet50'
    }

    params['waterbirds'] = {
        'n_groups': 4,
        # 'wd':[0.0001, 0.1, 1],
        'wd':[0.0001, 1],
        'lr': {
            0.0001: 1e-3,
            0.1: 1e-4,
            1: 1e-5
        },
        'bs': 128,
        'n_epochs': 300,
        'adjusted_wd': 1,
        'adj_list': [0, 1, 2, 3, 4, 5],
        'command': '-s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50'
    }

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
        'command': '-s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert'
    }

    for dataset in params.keys():
        for wd in params[dataset]['wd']:
            for opt_type in ['ERM', 'DRO']:
                lr = params[dataset]['lr'][wd]
                bs = params[dataset]['bs']
                n_epochs = params[dataset]['n_epochs']
                command = params[dataset]['command']
                adj = 0

                python_str = f'python3 src/run_expt.py {command} --weight_decay {wd} --lr {lr} --batch_size {bs} --n_epochs {n_epochs} --save_step 1000 --save_best --save_last'

                bundle_name = f'{dataset}_{opt_type}_wd-{wd}_lr-{lr}'

                if opt_type == 'reweight':
                    python_str += " --reweight_groups"
                elif opt_type == 'DRO':
                    python_str += (
                        " --reweight_groups --robust"
                        " --alpha 0.01 --gamma 0.1"
                        f" --generalization_adjustment {adj}"
                    )
                    bundle_name += f'_adj-{adj}'

                cl_cmd_str = f'cl run -n {bundle_name} --request-docker-image pangwei/group_dro:codalab --request-queue nlp --request-gpus 1 --request-memory 32g --request-network :src :{dataset} "{python_str}"'

                print(cl_cmd_str)
    # call(call_str,shell=True)

if __name__=='__main__':
    main()
