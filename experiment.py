import argparse

import numpy as np
import sys

from dgl import load_graphs

sys.path.append("..")
import build_dataset
import torch
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader
from model import collate_molgraphs, EarlyStopping, run_a_train_epoch_heterogeneous, \
    run_an_eval_epoch_heterogeneous, set_random_seed, MolP, pos_weight
import os
import time
import pandas as pd
start = time.time()
import os


args = {}
args['device'] = f"cuda:1"

args['atom_data_field'] = 'atom'
args['bond_data_field'] = 'etype'
args['classification_metric_name'] = 'roc_auc'
args['regression_metric_name'] = 'r2'
# model parameter
# model parameter
args['num_epochs'] = 500
args['patience'] = 50
args['batch_size'] = 128
args['mode'] = 'higher'
args['in_feats'] = 40

args['hidden_feats'] = [128, 64, 32, 32]

args['classifier_hidden_feats'] = 128
args['drop_out'] = 0.3
args['lr'] = 3
args['weight_decay'] = 5
args['loop'] = True
args['num_layers'] = 3
args['times'] = 10
args['pooling'] = 'sum'  # max/avg/sum

args['all_task_list'] ={'ADMET': ['F20', 'F30', 'HIA', 'PAMPA_NCATS', 'Pgp-inhibitor', 'Pgp-substrate',
                              'BBB', 'Half_Life', 'T0.5', 'CYP1A2 inhibitor', 'CYP1A2 substrate', 'CYP2C19 inhibitor', 'CYP2C19 substrate',
                              'CYP2C9 inhibitor', 'CYP2C9 substrate', 'CYP2D6 inhibitor', 'CYP2D6 substrate', 'CYP3A4 inhibitor',
                              'CYP3A4 substrate', 'AMES', 'Carcinogenicity', 'ClinTox', 'DILI', 'Eye Corrosion', 'Eye Irritation',
                              'FDAMDD', 'H-HT-2', 'hERG',
                              'NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma', 'Respiratory', 'ROA',
                              'Skin Sensitization', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53','PAMPA_NCATS','Pgp_Broccatelli','HIA_Hou','Caco2',  'Lipophilicity_AstraZeneca', 'MDCK Permeability',
                          'Solubility_AqSolDB', 'Fu', 'PPB', 'VD', 'CL',
                          'BCF', 'IGC50', 'LC50', 'LC50DM','Caco2_Wang','HydrationFreeEnergy_FreeSolv'
                ]}#change
args['classification_tasks']=['F20', 'F30', 'HIA', 'PAMPA_NCATS', 'Pgp-inhibitor', 'Pgp-substrate',
                              'BBB', 'Half_Life', 'T0.5', 'CYP1A2 inhibitor', 'CYP1A2 substrate', 'CYP2C19 inhibitor', 'CYP2C19 substrate',
                              'CYP2C9 inhibitor', 'CYP2C9 substrate', 'CYP2D6 inhibitor', 'CYP2D6 substrate', 'CYP3A4 inhibitor',
                              'CYP3A4 substrate', 'AMES', 'Carcinogenicity', 'ClinTox', 'DILI', 'Eye Corrosion', 'Eye Irritation',
                              'FDAMDD', 'H-HT-2', 'hERG',
                              'NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma', 'Respiratory', 'ROA',
                              'Skin Sensitization', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53','PAMPA_NCATS','Pgp_Broccatelli','HIA_Hou',]

args['regression_tasks']=['Caco2',  'Lipophilicity_AstraZeneca', 'MDCK Permeability',
                          'Solubility_AqSolDB', 'Fu', 'PPB', 'VD', 'CL',
                          'BCF', 'IGC50', 'LC50', 'LC50DM','Caco2_Wang','HydrationFreeEnergy_FreeSolv']




for dataset in args['all_task_list']:
    try:
        # task name (model name)
        args['task_name'] = dataset # change
        args['data_name'] = dataset  # change
        args['select_task_list'] =args['all_task_list'][dataset]
        args['select_task_category'] = []
        args['classification_num'] = 0
        args['regression_num'] = 0
        args['select_task_list'] = sorted(
            args['select_task_list'],
            key=lambda task: (task not in args['classification_tasks'], task not in args['regression_tasks'])
        )
        args['bin_path'] = []
        args['group_path'] = []
        for task in args['select_task_list']:
            args['bin_path'].append(f"../data/{task}_tdc.bin")
            args['group_path'].append(f"../data/{task}_group_tdc.npz")

            if task in args['classification_tasks']:
                args['classification_num'] = args['classification_num'] + 1
                args['select_task_category'].append("classification")
            if task in args['regression_tasks']:
                args['regression_num'] = args['regression_num'] + 1
                args['select_task_category'].append("regression")
        # generate classification_num
        if args['classification_num'] != 0 and args['regression_num'] != 0:
            args['task_class'] = 'classification_regression'
        if args['classification_num'] != 0 and args['regression_num'] == 0:
            args['task_class'] = 'classification'
        if args['classification_num'] == 0 and args['regression_num'] != 0:
            args['task_class'] = 'regression'
        print('Classification task:{}, Regression Task:{}'.format(args['classification_num'], args['regression_num']))




        result_pd = pd.DataFrame(columns=args['select_task_list']+['group'] + args['select_task_list']+['group']
                                 + args['select_task_list']+['group'])
        all_times_train_result = []
        all_times_val_result = []
        all_times_test_result = []
        for time_id in range(args['times']):
            set_random_seed(3407+time_id)
            one_time_train_result = []
            one_time_val_result = []
            one_time_test_result = []
            print('***************************************************************************************************')
            print('{}, {}/{} time'.format(args['task_name'], time_id+1, args['times']))
            print('***************************************************************************************************')
            train_set, val_set, test_set, task_number = build_dataset.load_graph_from_csv_bin_for_splited(
                bin_paths=args['bin_path'],
                group_paths=args['group_path'],
                select_task_num=len(args['select_task_list']),
                batch_size=args['batch_size'],
                category_list=args['select_task_category']
            )
            print("Molecule graph generation is complete !")

            model = MolP(in_feats=args['in_feats'], hidden_feats=args['hidden_feats'],
                        n_tasks=task_number, classifier_hidden_feats=args['classifier_hidden_feats'],
                        dropout=args['drop_out'], loop=args['loop'], n_layers=args['num_layers'])

            optimizer = AdamW(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])
            model = model.to(args['device'])

            stopper = EarlyStopping(pretrained_model="ADMET_early_stop_atten.pth",patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
            train_loader = DataLoader(dataset=train_set,
                                      batch_size=args['batch_size'],
                                      shuffle=True,
                                      collate_fn=collate_molgraphs, num_workers=0,)

            val_loader = DataLoader(dataset=val_set,
                                    batch_size=args['batch_size'],
                                    shuffle=False,
                                    collate_fn=collate_molgraphs)

            test_loader = DataLoader(dataset=test_set,
                                     batch_size=args['batch_size'],
                                     shuffle=False,
                                     collate_fn=collate_molgraphs, )
            pos_weight_np = pos_weight(train_set, classification_num=args['classification_num'])
            loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none',
                                                          pos_weight=pos_weight_np.to(args['device']))
            loss_criterion_r = torch.nn.MSELoss(reduction='none')

            for epoch in range(args['num_epochs']):
                # Train

                run_a_train_epoch_heterogeneous(args, epoch, model, train_loader, loss_criterion_c, loss_criterion_r, optimizer)

                # Validation and early stop

                validation_result = run_an_eval_epoch_heterogeneous(args, model, val_loader)

                val_score = np.mean(validation_result)
                early_stop = stopper.step(val_score, model)
                print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
                    epoch + 1, args['num_epochs'],
                    val_score,  stopper.best_score)+' validation result:', validation_result)
                if early_stop:
                    break
            stopper.load_checkpoint(model)
            test_score = run_an_eval_epoch_heterogeneous(args, model, test_loader)
            train_score = run_an_eval_epoch_heterogeneous(args, model, train_loader)
            val_score = run_an_eval_epoch_heterogeneous(args, model, val_loader)
            result = test_score + ['test'] + test_score + ['test']+ val_score + ['val']
            result_pd.loc[time_id] = result

            print('********************************{}, {}_times_result*******************************'.format(args['task_name'], time_id+1))
            print("training_result:", train_score)
            print("val_result:", val_score)
            print("test_result:", test_score)

        result_pd.to_csv('../result/atten_fusion/'+args['task_name']+'_result_atten.csv', index=None)
        elapsed = (time.time() - start)
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        print("Time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))
    except Exception as e:
        print(e)
        continue












