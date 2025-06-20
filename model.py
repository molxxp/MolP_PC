from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes,max_nodes,mean_nodes
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import pandas as pd


from dgl.nn import GATConv
import math

# from new_3d_feature import get_3D_GAT, GraphGather, flatten
from Extract3DFeature import Extract3DFeatures
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem
from ADMET_utils import pubchemfp
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight = return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self,feats):
        feat_list = []
        atom_list = []


        for i in range(self.task_num):

            feats_weight=self.atom_weighting_specific[i](feats)
            specific_feats=feats_weight*feats
            feat_list.append(specific_feats)
            atom_list.append(feats_weight)

        feat_data = torch.stack(feat_list)
        feats_weight = self.shared_weighting(feats)
        shared_feats = feats_weight * feats
        if self.attention:
            if self.return_weight:
                return feat_data, atom_list
            else:
                return feat_data
        else:

            return shared_feats

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )





class Attention_fusion(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v,weight_3d=2.0):
        super(Attention_fusion, self).__init__()

        self.q = nn.Sequential(nn.LayerNorm(dim_k), nn.Linear(input_dim, dim_k), nn.LayerNorm(dim_k), nn.Dropout(0.2),)
        self.k = nn.Sequential(nn.LayerNorm(dim_k), nn.Linear(input_dim, dim_k),nn.LayerNorm(dim_k),nn.Dropout(0.2),)
        self.v = nn.Sequential(nn.Linear(input_dim, dim_v),nn.LayerNorm(dim_v), nn.ReLU())

        self._norm_fact = 1 / math.sqrt(dim_k)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        temperature = 2
        score= self.bn(torch.bmm(Q, K.permute(0, 2, 1)) * self._norm_fact)
        atten = nn.Softmax(-1)(score/temperature)


        output = torch.bmm(atten, V)

        out = torch.mean(output, dim=1)  # (B, D)

        return out





class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels=64*21, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5,num_bases=None):
        super(RGCNLayer, self).__init__()

        self.activation = activation
        self.bn = batchnorm
        self.residual = residual
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                               num_bases=num_bases, bias=True, activation=activation,
                                               self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual

        if residual and in_feats != out_feats:
            self.res_connection = nn.Linear(in_feats, out_feats, bias=False)


        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

        self.dropout = nn.Dropout(rgcn_drop_out)
    def forward(self, bg, node_feats, etype, norm=None):
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)

        if self.residual:
            if hasattr(self, 'res_connection'):
                res_feats = self.res_connection(node_feats)
            else:
                res_feats = node_feats
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        new_feats = self.activation(new_feats)
        new_feats = self.dropout(new_feats)
        return new_feats
class GAT(nn.Module):
    def __init__(self,
                 n_layers,
                 in_feats,
                 n_hidden,
                 n_classes,
                 heads,
                 activation,
                 in_drop,
                 at_drop,
                 negative_slope,
                 ):
        super(GAT, self).__init__()
        self.num_layers = n_layers
        self.activation = activation

        self.gat_layers = nn.ModuleList()

        self.gat_layers.append(GATConv(in_feats, n_hidden, heads[0],
                                       in_drop, at_drop, negative_slope, activation=self.activation, allow_zero_in_degree=True))

        for l in range(1, n_layers):
            self.gat_layers.append(GATConv(n_hidden * heads[l-1], n_hidden, heads[l],
                                           in_drop, at_drop, negative_slope, activation=self.activation, allow_zero_in_degree=True))

        self.gat_layers.append(GATConv(n_hidden * heads[-2], n_classes, heads[-1],
                                       in_drop, at_drop, negative_slope, activation=None, allow_zero_in_degree=True))
        self.norms = nn.ModuleList([nn.LayerNorm(n_hidden * heads[i]) for i in range(n_layers)])

    def forward(self, g, inputs):
        h = inputs

        for l in range(self.num_layers):
            h_new = self.gat_layers[l](g, h).flatten(1)
            h_new = self.norms[l](h_new)
            h_new = F.leaky_relu(h_new, negative_slope=0.2)
            h_new = F.dropout(h_new, p=0.2)
            h = h_new + h if h.shape == h_new.shape else h_new

        logits = self.gat_layers[-1](g, h).sum(1)

        return logits



class FPN(nn.Module):
    def __init__(self, hidden_feats ,fp_type="mixed",fp_changebit = None ):
        super(FPN, self).__init__()
        self.fp_2_dim = 512
        self.hidden_dim = hidden_feats
        self.fp_type = fp_type
        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024
        self.fp_changebit = fp_changebit
        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(self.fp_2_dim)  # BatchNorm

    def forward(self, smiles,device):

        fp_list = []
        for i, one in enumerate(smiles):
            fp = []
            mol = Chem.MolFromSmiles(one)
            mol = Chem.RemoveHs(mol)
            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
                fp_pubcfp = pubchemfp.GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)
        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:, self.fp_changebit - 1] = np.ones(fp_list[:, self.fp_changebit - 1].shape)
            fp_list.tolist()
        fp_list = torch.tensor(np.array(fp_list)).float().to(device)
        fpn_out = self.fc1(fp_list)
        fpn_out = self.batch_norm1(fpn_out)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        # del fp_list
        return fpn_out
class MolP(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks,n_layers, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=256, dropout=0.2, pooling="sum",ftype="mixed",loop=False):
        super(MolP, self).__init__()
        self.in_feats = in_feats
        self.gnn_layers = nn.ModuleList()
        self.out_feats =  hidden_feats[-2]
        self.task_num = n_tasks
        self.return_weight = return_weight
        # 返回的是task_num个TFP列表
        self.return_mol_embedding = return_mol_embedding
        # 1D
        self.fpn = FPN(hidden_feats=hidden_feats[-3], fp_type=ftype)

        #2D
        for i in range(len(hidden_feats)):
            self.gnn_layers.append(RGCNLayer(in_feats, hidden_feats[i], loop=loop, rgcn_drop_out=0.2))
            in_feats=hidden_feats[i]
        self.GATattention = GAT(n_layers=n_layers, in_feats=self.in_feats, n_hidden=self.out_feats,n_classes=self.out_feats,
                                heads=([n_layers] * n_layers) + [1], activation=F.gelu, in_drop=0.2  , at_drop=0.5, negative_slope=0.2)


        #3D
        self.extra_3d = Extract3DFeatures(node_feats_dim=self.in_feats, edge_feats_dim=1,
                                          node_mlp_hidden_dim=hidden_feats[-3],
                                          edge_mlp_hidden_dim=hidden_feats[-3], edge_attr_dim=3, dropout=dropout)
        self.get_3d_gat = GAT(n_layers=n_layers, in_feats=self.in_feats, n_hidden=hidden_feats[-3], n_classes=hidden_feats[-3],
                              heads=([n_layers] * n_layers) + [1], activation=F.gelu, in_drop=0.2, at_drop=0.5, negative_slope=0.2)

        self.atten_fusion=Attention_fusion(hidden_feats[-3],hidden_feats[-3],hidden_feats[-3])

        self.weighted_sum_readout = WeightAndSum(hidden_feats[-3], self.task_num, return_weight=self.return_weight)

        self.fc_layers1 = nn.ModuleList(
            [self.fc_layer(dropout, hidden_feats[-3], classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])


    def forward(self,smiles,pos_g, bg, node_feats, etype, device='cpu',norm=None):

        #3D
        pos_node_feats=pos_g.ndata['atom'].float()
        mask = pos_g.ndata['mask'].float().unsqueeze(1)
        updated_nodes=self.extra_3d(pos_g,pos_node_feats,pos_g.ndata['pos'].float(),edge_attr=pos_g.edata['etype'].float())
        new_feats=self.get_3d_gat(pos_g,updated_nodes)*mask
        three_feats=self.flatten(pos_g, new_feats)
        #2D
        node_feats_fc = node_feats
        for gnn in self.gnn_layers:
            node_feats_fc = gnn(bg, node_feats_fc, etype, norm)
        atten_feats = self.GATattention(bg, node_feats)
        atten_feats = torch.cat([node_feats_fc, atten_feats], dim=-1)
        two_feats = self.flatten(bg, atten_feats)
        #1D
        fp_feats=self.fpn(smiles,device)
        #fusion
        feats_out=torch.stack([fp_feats,two_feats,three_feats],dim=1)
        feats=self.atten_fusion(feats_out)

        if self.return_weight:
            feats_list, task_weight_list = self.weighted_sum_readout(feats)
        else:
            feats_list = self.weighted_sum_readout(feats)
        for i in range(self.task_num):
            mol_feats = feats_list[i]#取TFP
            mol_feats = torch.cat([mol_feats], dim=1)

            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)
        # generate toxicity fingerprints
        if self.return_mol_embedding:
            return "return_mol_embedding"

        else:
            # generate atom weight and atom feats
            if self.return_weight:

                return prediction_all, task_weight_list,


            return prediction_all

    def flatten(self,bg,feats,pooling="sum"):
        with bg.local_scope():
            bg.ndata['h'] = feats
            if pooling=='sum':
                specific_feats = sum_nodes(bg,'h')
            elif pooling=="max":
                specific_feats = max_nodes(bg, 'h')
            elif pooling=="avg":
                specific_feats = mean_nodes(bg, 'h')
            return specific_feats


    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )
    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pos_weight(train_set, classification_num):
    smiles, pos_g, graphs, labels, mask = map(list, zip(*train_set))


    labels = np.array(labels)
    task_pos_weight_list = []
    for task in range(classification_num):
        num_pos = 0
        num_impos = 0
        for i in labels[:, task]:
            if i == 1:
                num_pos = num_pos + 1
            if i == 0:
                num_impos = num_impos + 1
        weight = num_impos / (num_pos+0.00000001)
        task_pos_weight_list.append(weight)
    task_pos_weight = torch.tensor(task_pos_weight_list)
    return task_pos_weight


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters更
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)，
        mask : float32 tensor
            Mask for indicating the existence of ground，
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes 分类
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        return y_pred, y_true

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())

        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        print(y_true.shape, y_pred.shape)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            print(task_y_true.shape, task_y_pred.shape)
            # scores.append(round(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item(), 4)))
            mse = F.mse_loss(task_y_pred, task_y_true)
            rmse_value = np.sqrt(mse.cpu().item())
            scores.append(round(rmse_value, 4))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)

            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'mae', 'roc_prc', 'r2', 'return_pred_true'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_recall_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()


def collate_molgraphs(data):
    smiles, pos_g, graphs, labels,mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    pos_bg = dgl.batch(pos_g)
    pos_bg.set_n_initializer(dgl.init.zero_initializer)
    pos_bg.set_e_initializer(dgl.init.zero_initializer)

    labels = torch.tensor(np.array(labels))
    mask = torch.tensor(np.array(mask))


    return smiles, pos_bg, bg,labels,mask
def run_a_train_epoch_heterogeneous(args, epoch, model, data_loader, loss_criterion_c, loss_criterion_r, optimizer, task_weight=None):
    model.train()
    train_meter_c = Meter()
    train_meter_r = Meter()
    if task_weight is not None:
        task_weight = task_weight.float().to(args['device'])
    loss_all=0


    for batch_id, batch_data in enumerate(data_loader):
        smiles, pos_g, bg, labels, mask = batch_data
        mask = mask.float().to(args['device'])
        pos_g=pos_g.to(args['device'])
        bg = bg.to(args['device'])
        atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
        bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
        logits = model(smiles,pos_g, bg, atom_feats, bond_feats,device=args['device'],norm=None).to(args['device'])

        labels = labels.type_as(logits).to(args['device'])
        # calculate loss according to different task class
        if args['task_class'] == 'classification_regression':
            # split classification and regression
            logits_c = logits[:,:args['classification_num']]
            labels_c = labels[:,:args['classification_num']]
            mask_c = mask[:,:args['classification_num']]

            logits_r = logits[:,args['classification_num']:]
            labels_r = labels[:,args['classification_num']:]
            mask_r = mask[:,args['classification_num']:]
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float()).mean() \
                       + (loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float()).mean()
            else:
                task_weight_c = task_weight[:args['classification_num']]
                task_weight_r = task_weight[args['classification_num']:]
                loss = (torch.mean(loss_criterion_c(logits_c, labels_c)*(mask_c != 0).float(), dim=0)*task_weight_c).mean() \
                       + (torch.mean(loss_criterion_r(logits_r, labels_r)*(mask_r != 0).float(), dim=0)*task_weight_r).mean()
            loss_all = loss_all + loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_c.update(logits_c, labels_c, mask_c)
            train_meter_r.update(logits_r, labels_r, mask_r)
            # del bg, mask, labels, atom_feats, bond_feats, loss, logits_c, logits_r, labels_c, labels_r, mask_c, mask_r,pos_g
        elif args['task_class'] == 'classification':
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_c(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_c(logits, labels) * (mask != 0).float(),dim=0)*task_weight).mean()
            loss_all = loss_all + loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_c.update(logits, labels, mask)

        else:
            # chose loss function according to task_weight
            if task_weight is None:
                loss = (loss_criterion_r(logits, labels)*(mask != 0).float()).mean()
            else:
                loss = (torch.mean(loss_criterion_r(logits, labels) * (mask != 0).float(), dim=0)*task_weight).mean()
            loss_all = loss_all + loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter_r.update(logits, labels, mask)

    if args['task_class'] == 'classification_regression':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']) +
                              train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f} ,loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], 'r2+auc', train_score,loss_all/len(data_loader)))
    elif args['task_class'] == 'classification':
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f} ,loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['classification_metric_name'], train_score,loss_all/len(data_loader)))
    else:
        train_score = np.mean(train_meter_r.compute_metric(args['regression_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f} ,loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['regression_metric_name'], train_score,loss_all/len(data_loader)))


def run_an_eval_epoch_heterogeneous(args, model, data_loader):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, pos_g, bg, labels, mask = batch_data
            mask = mask.float().to(args['device'])
            pos_g = pos_g.to(args['device'])
            bg = bg.to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits = model(smiles, pos_g, bg, atom_feats, bond_feats, device=args['device'], norm=None).to(args['device'])
            labels = labels.type_as(logits).to(args['device'])
            if args['task_class'] == 'classification_regression':
                # split classification and regression
                logits_c = logits[:, :args['classification_num']]
                labels_c = labels[:, :args['classification_num']]
                mask_c = mask[:, :args['classification_num']]
                logits_r = logits[:, args['classification_num']:]
                labels_r = labels[:, args['classification_num']:]
                mask_r = mask[:, args['classification_num']:]
                # Mask non-existing labels
                eval_meter_c.update(logits_c, labels_c, mask_c)
                eval_meter_r.update(logits_r, labels_r, mask_r)
            elif args['task_class'] == 'classification':
                # Mask non-existing labels
                eval_meter_c.update(logits, labels, mask)
            else:
                # Mask non-existing labels
                eval_meter_r.update(logits, labels, mask)


        if args['task_class'] == 'classification_regression':
            return eval_meter_c.compute_metric(args['classification_metric_name']) + \
                   eval_meter_r.compute_metric(args['regression_metric_name'])
        elif args['task_class'] == 'classification':
            return eval_meter_c.compute_metric(args['classification_metric_name'])
        else:
            return eval_meter_r.compute_metric(args['regression_metric_name'])

class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            self.task_name = task_name

            filename = '../model/atten_fusion/{}_early_stop_atten.pth'.format(task_name)


        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience#50
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''


        torch.save(model, self.filename)

    def load_checkpoint(self, model):

        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename,map_location=torch.device('cpu')).state_dict())





    def load_pretrained_model(self, model):
        pretrained_parameters=[param for param in model.parameters()]
        if torch.cuda.is_available():
            pretrained_model = torch.load('../model/all/'+self.pretrained_model,map_location=torch.device('cpu'))
        else:
            pretrained_model = torch.load('../model/all/'+self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()

        pretrained_dict={k: v for k, v in pretrained_model.state_dict().items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)





