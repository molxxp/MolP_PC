import argparse
import os

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops , MolFromSmiles
import torch
from dgl import DGLGraph
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import train_test_split
import re
from dgl.data.graph_serialize import save_graphs, load_graphs
from multiprocessing import Pool
import pubchemfp


def split_dataset_according_index(dataset, train_index, val_index, test_index, data_type='np'):
    if data_type == 'pd':
        return pd.DataFrame(dataset[train_index]), pd.DataFrame(dataset[val_index]), pd.DataFrame(dataset[test_index])
    if data_type == 'np':
        return dataset[train_index], dataset[val_index], dataset[test_index]

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        if 'other' in allowable_set:
            x = 'other'
        else:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom):
    atom_type = one_of_k_encoding(atom.GetSymbol(),
                                  ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At',
                                   'other']) + \
                one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
                [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(),
                                      [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                       Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                       Chem.rdchem.HybridizationType.SP3D2, 'other']) + \
                [atom.GetIsAromatic()] + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    try:
        atom_type = atom_type + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        atom_type = atom_type + [False, False] + [atom.HasProp('_ChiralityPossible')]
    atom_type = np.array(atom_type)
    atom_type = atom_type.astype(int)
    return atom_type


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        else:
            if atompair == allowable_set[-1]:
                x = allowable_set[-1]
            else:
                continue
    return [x == s for s in allowable_set]


def etype_features(bond, use_chirality=True, atompair=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    if atompair == True:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats_5 = one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )
        for i, m in enumerate(bond_feats_5):
            if m == True:
                e = i
        index = index + e * 64
    return index


def construct_bigraph_from_smiles(smiles):
    try:
        g = DGLGraph()
        # Add nodes
        mol = MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        num_atoms = mol.GetNumAtoms()


        # num_atoms = mol.GetNumAtoms()
        g.add_nodes(num_atoms)

        atoms_feature_all = []
        for atom_index, atom in enumerate(mol.GetAtoms()):
            atom_feature = atom_features(atom).tolist()
            atoms_feature_all.append(atom_feature)
        g.ndata["atom"] = torch.tensor(atoms_feature_all)

        # Add edges
        src_list = []
        dst_list = []
        etype_feature_all = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            etype_feature = etype_features(bond)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            src_list.extend([u, v])
            dst_list.extend([v, u])
            etype_feature_all.append(etype_feature)
            etype_feature_all.append(etype_feature)

        g.add_edges(src_list, dst_list)
        normal_all = []
        for i in etype_feature_all:
            normal = etype_feature_all.count(i) / len(etype_feature_all)
            normal = round(normal, 1)
            normal_all.append(normal)

        g.edata["etype"] = torch.tensor(etype_feature_all)
        g.edata["normal"] = torch.tensor(normal_all)


        coord = get_smiles_coord(smiles)
        g.ndata["pos"] = coord
        return g
    except Exception as e:
        raise ValueError(f"{smiles} failed to transform: {e}")


def build_mask(labels_list, mask_value=100):
    mask = []
    for i in labels_list:
        if i == mask_value:
            mask.append(0)
        else:
            mask.append(1)

    return mask

def smiles_match(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    inchi1 = Chem.MolToInchiKey(mol1)
    inchi2 = Chem.MolToInchiKey(mol2)
    return inchi1 == inchi2

def get_smiles_coord(smile):
    mol = Chem.MolFromSmiles(smile)
    num_atoms = mol.GetNumAtoms()
    coords_tensor = torch.zeros((num_atoms, 3), dtype=torch.float32)

    try:
        mol_3d = Chem.AddHs(mol)
        # AllChem.EmbedMolecule(mol_3d, randomSeed=1)

        success = AllChem.EmbedMultipleConfs(mol_3d, maxAttempts=10, randomSeed=12345)
        if not success:
            raise ValueError("Failed to generate 3D conformers.")

        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=100)

        conf = mol_3d.GetConformer()
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            coords_tensor[i, :] = torch.tensor([pos.x, pos.y, pos.z], dtype=torch.float32)
        return coords_tensor
    except Exception as e:
        raise ValueError(f"Error generating 3D coordinates for SMILES '{smile}': {e}")

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = rdMolStandardize.Cleanup(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None
def multi_task_build_dataset(dataset_smiles, labels_list, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_list]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]

    # 遍历SMILES列表
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        try:
            g= construct_bigraph_from_smiles(smiles)
            mask = build_mask(labels.loc[i], mask_value=123456)  #
            molecule = [smiles, g,labels.loc[i], mask,
                        split_index.loc[i]]
            dataset_gnn.append(molecule)
            print('{}/{} molecule is transformed!'.format(i + 1, molecule_number))
        except:
            # print('{} is transformed failed!'.format(smiles))
            molecule_number = molecule_number - 1
            failed_molecule.append(smiles)
            print('{}/{} molecule is failed!'.format(i + 1, molecule_number))
    print('{}({}) is transformed failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn

def load_graph_from_csv_bin_for_splited(
        bin_paths='example.bin',
        group_paths='example.npz',
        select_task_num=1,
        batch_size=1,
        category_list=None

):
    from collections import defaultdict
    all_smiles,all_group,all_graphs,all_labels,all_masks,all_pos_g=[],[],[],[],[],[]
    labels_size = 0
    masks_size = 0
    smiles_dict=defaultdict(lambda:{'labels':[],'masks':[],'pos_g':[],'bg':[],})
    for task_id, (group_path, bin_path,category) in enumerate(zip(group_paths, bin_paths,category_list)):
        data = np.load(group_path, allow_pickle=True)
        smiles_list = data['smiles']
        group = data['group']

        graphs, detailed_information = load_graphs(bin_path)

        labels = detailed_information['labels']
        mask = detailed_information['mask']



        graphs=np.array(graphs)

        all_smiles.extend(smiles_list)
        all_group.extend(group)
        all_graphs.extend(graphs)

        all_labels.append(labels)
        all_masks.append(mask)
        labels_size += labels.size(0)
        masks_size += mask.size(0)
    task_number=len(all_labels)
    num=0
    num_mask=0

    for i in range(len(all_labels)):
        batch_size=len(all_labels[i])
        full_pad=123456.0
        begin=torch.full((num, 1), full_pad, dtype=torch.float64)
        num+=batch_size
        end=torch.full((labels_size-num, 1), full_pad, dtype=torch.float64)
        all_labels[i] = torch.cat((begin,all_labels[i],end),dim=0)
        full_pad = 0
        begin = torch.full((num_mask, 1), full_pad, dtype=torch.float64)
        num_mask += batch_size
        end = torch.full((labels_size - num_mask, 1), full_pad, dtype=torch.float64)
        all_masks[i] = torch.cat((begin,all_masks[i],end),dim=0)

    all_labels=torch.cat(all_labels,dim=1)
    all_masks=torch.cat(all_masks,dim=1)

    train_index = []
    val_index = []
    test_index = []

    max_node_count = max([g.num_nodes() for g in all_graphs])
    for g in all_graphs:

        pos_g = DGLGraph()
        features = g.ndata['atom']
        pos = g.ndata['pos']
        N = g.num_nodes()
        node_mask = torch.zeros(max_node_count)


        node_mask[:N] = 1


        if N < max_node_count:

            padding_size = max_node_count - features.shape[0]


            features_matrix = torch.cat((features, torch.zeros(padding_size, features.shape[1])), dim=0,)
            pos_matrix = torch.cat((pos, torch.zeros(padding_size, pos.shape[1])), dim=0)
        else:

            features_matrix = features[:max_node_count]
            pos_matrix = pos[:max_node_count]

        pos_g.add_nodes(max_node_count)

        pos_g.ndata['atom'] = features_matrix.float()
        pos_g.ndata['pos'] = pos_matrix.float()
        pos_g.add_edges(g.edges()[0], g.edges()[1])
        pos_g.edata["etype"]=g.edata["normal"].float()

        pos_g.ndata['mask'] = node_mask.float()
        all_pos_g.append(pos_g)


    for i in range(labels_size):
        smile=all_smiles[i]
        group=all_group[i]
        key=(smile,group)
        smiles_dict[key]['labels'].append(all_labels[i])
        smiles_dict[key]['masks'].append(all_masks[i])
        smiles_dict[key]['pos_g']=all_pos_g[i]
        smiles_dict[key]['bg']=all_graphs[i]
    all_smiles, all_group, all_graphs, all_pos_g,all_labels,all_masks=[],[],[],[],[],[]
    for (smiles, group), data in smiles_dict.items():

        if len(data['labels'])==task_number:
            result = []
            result_mask=[]
            for col in range(task_number):

                column_values = [tensor[col].item() for tensor in data['labels']]
                mask_values = [tensor[col].item() for tensor in data['masks']]

                if all(value == 123456.0 for value in column_values):
                    result.append(123456.0)
                else:

                    column_values_filtered = [value for value in column_values if value != 123456.0]
                    result.append(sum(column_values_filtered))
                if all(value == 0 for value in mask_values):
                    result_mask.append(0)
                else:
                    # column_values_filtered = [value for value in column_values if value != 123456]
                    result_mask.append(sum(mask_values))
            data['labels'].append(torch.tensor(result))
            data['masks'].append(torch.tensor(result_mask))
        all_smiles.append(smiles)
        all_group.append(group)
        all_graphs.append(data['bg'])
        all_pos_g.append(data['pos_g'])
        all_labels.append(data['labels'])
        all_masks.append(data['masks'])
    all_smiles, all_group, all_graphs, all_pos_g = np.array(all_smiles), np.array(all_group), np.array(
        all_graphs), np.array(all_pos_g)

    tensors = [item[0] for item in all_labels]


    labels= torch.stack(tensors)
    tensors = [item[0] for item in all_masks]

    mask = torch.stack(tensors)
    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    not_use_index = []
    for index, notuse in enumerate(notuse_mask):
        if notuse == 0:
            not_use_index.append(index)

    for index, group_index in enumerate(all_group):
        if group_index == 'training' and index not in not_use_index:
            train_index.append(index)
        if group_index == 'val' and index not in not_use_index:
            val_index.append(index)
        if group_index == 'test' and index not in not_use_index:
            test_index.append(index)
    target_index = val_index if len(val_index) > len(test_index) else test_index
    if target_index:
        train_index.append(target_index.pop())
    else:
        val_index.append(train_index.pop())

    train_smiles, val_smiles, test_smiles = split_dataset_according_index(all_smiles, train_index, val_index, test_index)

    train_labels, val_labels, test_labels = split_dataset_according_index(labels.numpy(), train_index, val_index,
                                                                          test_index, data_type='pd')
    train_mask, val_mask, test_mask = split_dataset_according_index(mask.numpy(), train_index, val_index, test_index,
                                                                    data_type='pd')
    train_graph, val_graph, test_graph = split_dataset_according_index(all_graphs, train_index, val_index, test_index,)
    train_pos_g, val_pos_g, test_pos_g = split_dataset_according_index(all_pos_g, train_index, val_index, test_index, )


    train_set = []
    val_set = []
    test_set = []

    for i in range(len(train_index)):
        molecule = [train_smiles[i], train_pos_g[i],train_graph[i], train_labels.values[i], train_mask.values[i],]
        train_set.append(molecule)

    for i in range(len(val_index)):
        molecule = [val_smiles[i], val_pos_g[i],val_graph[i], val_labels.values[i], val_mask.values[i], ]
        val_set.append(molecule)

    for i in range(len(test_index)):
        molecule = [test_smiles[i],test_pos_g[i], test_graph[i], test_labels.values[i], test_mask.values[i], ]
        test_set.append(molecule)

    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number

def load_graph_from_csv_bin_for_splited_multi(
        bin_path='example.bin',
        group_path='example.npz',
        batch_size=32,
        select_task_index=None):
    data = np.load(group_path, allow_pickle=True)
    smiles = data['smiles']
    group = data['group']

    graphs, detailed_information = load_graphs(bin_path)
    labels = detailed_information['labels']

    mask = detailed_information['mask']
    max_node_count = max([g.num_nodes() for g in graphs])
    if select_task_index is not None:
        labels = labels[:, select_task_index]
        mask = mask[:, select_task_index]
    # calculate not_use index
    notuse_mask = torch.mean(mask.float(), 1).numpy().tolist()
    not_use_index = []
    for index, notuse in enumerate(notuse_mask):
        if notuse == 0:
            not_use_index.append(index)
    train_index = []
    val_index = []
    test_index = []
    for index, group_index in enumerate(group):
        if group_index == 'training' and index not in not_use_index:
            train_index.append(index)
        if group_index == 'valid' and index not in not_use_index:
            val_index.append(index)
        if group_index == 'val' and index not in not_use_index:
            val_index.append(index)
        if group_index == 'test' and index not in not_use_index:
            test_index.append(index)

    pos_g_list = []
    for g in graphs:

        pos_g = DGLGraph()
        features = g.ndata['atom']
        pos = g.ndata['pos']
        N = g.num_nodes()
        node_mask = torch.zeros(max_node_count)
        node_mask[:N] = 1
        if N < max_node_count:

            padding_size = max_node_count - features.shape[0]


            features_matrix = torch.cat((features, torch.zeros(padding_size, features.shape[1])), dim=0, )
            pos_matrix = torch.cat((pos, torch.zeros(padding_size, pos.shape[1])), dim=0)
        else:

            features_matrix = features[:max_node_count]
            pos_matrix = pos[:max_node_count]

        pos_g.add_nodes(max_node_count)
        pos_g.ndata['atom'] = features_matrix.float()
        pos_g.ndata['pos'] = pos_matrix.float()
        pos_g.add_edges(g.edges()[0], g.edges()[1])
        pos_g.edata["etype"] = g.edata["normal"].float()
        pos_g.ndata['mask'] = node_mask.float()
        pos_g_list.append(pos_g)

    graphs_np = np.array(graphs)
    pos_g = np.array(pos_g_list)
    train_smiles, val_smiles, test_smiles = split_dataset_according_index(smiles, train_index, val_index, test_index)

    train_labels, val_labels, test_labels = split_dataset_according_index(labels.numpy(), train_index, val_index,
                                                                          test_index, data_type='pd')
    train_mask, val_mask, test_mask = split_dataset_according_index(mask.numpy(), train_index, val_index, test_index,
                                                                    data_type='pd')
    train_graph, val_graph, test_graph = split_dataset_according_index(graphs_np, train_index, val_index, test_index)

    train_pos_g, val_pos_g, test_pos_g = split_dataset_according_index(pos_g, train_index, val_index, test_index, )

    task_number =train_labels.shape[1]

    train_set = []
    val_set = []
    test_set = []

    for i in range(len(train_index)):
        molecule = [train_smiles[i], train_pos_g[i], train_graph[i], train_labels.values[i], train_mask.values[i], ]
        train_set.append(molecule)

    for i in range(len(val_index)):
        molecule = [val_smiles[i], val_pos_g[i], val_graph[i], val_labels.values[i], val_mask.values[i], ]
        val_set.append(molecule)

    for i in range(len(test_index)):
        molecule = [test_smiles[i], test_pos_g[i], test_graph[i], test_labels.values[i], test_mask.values[i], ]
        test_set.append(molecule)

    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number

def process_file(args):
    try:
        path,i=args
        if i.endswith('.csv'):
            i = i.split('.csv')[0]
        else:
            i = i
        data_path = path + i + '.csv'
        save_path = path + i + '_tdc.bin'
        datanpz_path = path + i + '_group_tdc.npz'
        data_origin = pd.read_csv(data_path)
        smiles_name = 'smiles'
        data_origin = data_origin.fillna(123456)  # 填充空值
        labels_list = [x for x in data_origin.columns if x not in ['smiles', 'group']]

        data_set_gnn = multi_task_build_dataset(dataset_smiles=data_origin, labels_list=labels_list,
                                                smiles_name=smiles_name)
        smiles, graphs, labels, mask, split_index = map(list, zip(*data_set_gnn))

        graph_labels = {
            'labels': torch.tensor(labels),
            'mask': torch.tensor(mask)
        }
        np.savez(datanpz_path, smiles=smiles, group=split_index)
        save_graphs(save_path, graphs, graph_labels)
        print(f'Molecules graph for {i} is saved!')
    except Exception as e:
        print(f' Error processing file {i}: {e}')
if __name__ == '__main__':
    # 创建解析命令行参数的解析器
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('path', type=str, help='The directory containing the data files')
    parser.add_argument('ls', nargs='+', help='List of file names to process')

    args = parser.parse_args()

    args_list = [(args.path, i) for i in args.ls]
    with Pool(processes=10) as pool:
        pool.map(process_file, args_list)


































