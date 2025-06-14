import os
from itertools import repeat
import pandas as pd
import json
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem
from rdkit import RDLogger
import rdkit.Chem as Chem

RDLogger.DisableLog('rdApp.*')

from FineMolTex.datasets.utils import mol_to_graph_data_obj_simple, get_positions, tree_decomp, brics_decomp, get_clique_mol, get_smiles
from FineMolTex.datasets.mol_bpe import Tokenizer

class PubChemSTM_Datasets_Only_SMILES(Dataset):
    def __init__(self, root, subset_size=None):
        self.root = root

        CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
        
        df = pd.read_csv(CID2SMILES_file)
        SMILES_list = df["SMILES"].tolist()
        SMILES_list = sorted(set(SMILES_list))
        
        self.SMILES_list = SMILES_list
        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return
    
    def __getitem__(self, index):
        SMILES = self.SMILES_list[index]
        return SMILES

    def __len__(self):
        return len(self.SMILES_list)


def mol_frag_collate(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    # keys follow node
    node_sum_keys = ["edge_index"]
    # keys follow frag
    frag_sum_keys = ["frag_edge_index", "map"]
    # no sum keys
    no_sum_keys = ["edge_attr",
                   "x",
                   "frag",
                   "text",
                   "frag_unique"]

    for key in node_sum_keys + frag_sum_keys + no_sum_keys:
        batch[key] = []


    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    cumsum_node = 0
    i_node = 0

    cumsum_frag = 0
    i_frag = 0

    for data in data_list:
        num_nodes = data.x.shape[0]

        num_frags = data.frag.shape[0]

        batch.node_batch_size.append(num_nodes)

        batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes,), i_node, dtype=torch.long))

        batch.frag_batch.append(torch.full((num_frags,), i_frag, dtype=torch.long))

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        for key in frag_sum_keys:
            item = data[key]
            item = item + cumsum_frag
            batch[key].append(item)

        for key in no_sum_keys:
            item = data[key]
            batch[key].append(item)


        cumsum_node += num_nodes
        i_node += 1

        cumsum_frag += num_frags
        i_frag += 1

    batch.x = torch.cat(batch.x, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    batch.frag = torch.cat(batch.frag, dim=0)
    batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    batch.frag_unique = torch.cat(batch.frag_unique, dim=0)
    batch.map = torch.cat(batch.map, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    batch.frag_batch_size = torch.tensor(batch.frag_batch_size)


    return batch.contiguous()

def mol_frag_collate_retri(data_list):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly."""

    batch = Data()
    # keys follow node
    node_sum_keys = ["edge_index"]
    # keys follow frag
    frag_sum_keys = ["frag_edge_index", "map"]
    # no sum keys
    no_sum_keys = ["edge_attr",
                   "x",
                   "frag",
                   "text",
                   "frag_unique",
                   "neg_text",
                    "neg_data"
                   ]

    for key in node_sum_keys + frag_sum_keys + no_sum_keys:
        batch[key] = []

    for i in range(len(data_list[0]['neg_data'])):
        batch['neg_data'].append([])

    for i in range(len(data_list[0]['neg_text'])):
        batch['neg_text'].append([])

    batch.node_batch_size = []
    batch.node_batch = []

    batch.frag_batch_size = []
    batch.frag_batch = []

    cumsum_node = 0
    i_node = 0

    cumsum_frag = 0
    i_frag = 0

    j = 0
    for data in data_list:
        num_nodes = data.x.shape[0]

        num_frags = data.frag.shape[0]

        batch.node_batch_size.append(num_nodes)

        batch.frag_batch_size.append(num_frags)

        batch.node_batch.append(torch.full((num_nodes,), i_node, dtype=torch.long))

        batch.frag_batch.append(torch.full((num_frags,), i_frag, dtype=torch.long))

        for key in node_sum_keys:
            item = data[key]
            item = item + cumsum_node
            batch[key].append(item)

        for key in frag_sum_keys:
            item = data[key]
            item = item + cumsum_frag
            batch[key].append(item)

        for key in no_sum_keys:
            if key == "neg_data":
                item = data[key]
                for k in range(len(item)):
                    batch[key][k].append(item[k])
            elif key == "neg_text":
                item = data[key]
                for k in range(len(item)):
                    batch[key][k].append(item[k])
            else:
                item = data[key]
                batch[key].append(item)

        cumsum_node += num_nodes
        i_node += 1

        cumsum_frag += num_frags
        i_frag += 1

    batch.neg_data = [mol_frag_collate(item) for item in batch.neg_data]
    batch.x = torch.cat(batch.x, dim=0)
    batch.edge_index = torch.cat(batch.edge_index, dim=-1)
    batch.edge_attr = torch.cat(batch.edge_attr, dim=0)
    batch.frag = torch.cat(batch.frag, dim=0)
    batch.frag_edge_index = torch.cat(batch.frag_edge_index, dim=-1)
    batch.frag_unique = torch.cat(batch.frag_unique, dim=0)
    batch.map = torch.cat(batch.map, dim=-1)
    # for key in keys:
    #     batch[key] = torch.cat(
    #         batch[key], dim=batch.cat_dim(key))
    batch.node_batch = torch.cat(batch.node_batch, dim=-1)
    batch.node_batch_size = torch.tensor(batch.node_batch_size)
    batch.frag_batch = torch.cat(batch.frag_batch, dim=-1)
    batch.frag_batch_size = torch.tensor(batch.frag_batch_size)


    return batch.contiguous()


class PubChemSTM_Datasets_Motif(InMemoryDataset):
    def __init__(self, root, datafile, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.datafile = "Map_"+ datafile
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/{}/CID_text_list.csv".format(self.datafile[:-3]))
        if not os.path.exists(os.path.join(self.root, "processed/{}".format(self.datafile[:-3]))):
            os.makedirs(os.path.join(self.root, "processed/{}".format(self.datafile[:-3])))
        self.CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")

        self.tokenizer = Tokenizer("../data/PubChemSTM_data/vocab3.txt")
        self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}

        super(PubChemSTM_Datasets_Motif, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        # return 'geometric_data_motif_processed97.pt'
        # return 'geometric_data_motif_mask11_processed.pt'

        print(self.datafile)
        return self.datafile

    def process(self):

        suppl = Chem.SDMolSupplier(self.SDF_file_path)
        CID2graph = {}
        valid = 0

        # count = 0
        for mol in tqdm(suppl):
            # if count == 100:
            #     break
            # count = count + 1
            try:
                CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
            except:
                continue
            CID = int(CID)
            graph = mol_to_graph_data_obj_simple(mol)
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
            graph.smiles = smiles
            try:
                tree = self.tokenizer(smiles)
            except:
                print("Unable to process SMILES:", smiles)
                continue

            # Manually consructing the fragment graph
            map = [0]*graph.num_nodes
            frag = [[0] for _ in range(len(tree.nodes))]
            frag_edge_index = [[],[]]

            try:
                for node_i in tree.nodes:
                    node = tree.get_node(node_i)
                    # for atom in node, set map
                    for atom_i in node.atom_mapping.keys():
                        map[atom_i] = node_i
                    # extend frag
                        frag[node_i][0] = self.vocab_dict[node.smiles]
                for src, dst in tree.edges:
                    # extend edge index
                    frag_edge_index[0].extend([src,dst])
                    frag_edge_index[1].extend([dst,src])
            except KeyError as e:
                print("Error in matching subgraphs", e)
                continue

            unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
            frag_unique = torch.zeros(800).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

            graph.map = torch.LongTensor(map)
            graph.frag = torch.LongTensor(frag)
            graph.frag_edge_index = torch.LongTensor(frag_edge_index)
            graph.frag_unique = frag_unique
            CID2graph[CID] = graph


        print("CID2graph", len(CID2graph))
        print(valid)

        with open(self.CID2text_file, "r") as f:
            CID2text_data = json.load(f)
        print("CID2data", len(CID2text_data))

        CID_list, graph_list, text_list = [], [], []
        missing = 0
        for CID, value_list in CID2text_data.items():
            CID = int(CID)
            if CID not in CID2graph:
                print("CID {} missing".format(CID))
                missing = missing + 1
                continue
            graph = CID2graph[CID]

            # clique = CID2clique[CID]
            for value in value_list:
                text_list.append(value)
                CID_list.append(CID)
                graph.text = value
                graph_list.append(graph)

        print("total missing: {}".format(missing))
        CID_text_df = pd.DataFrame({"CID": CID_list, "text": text_list})
        CID_text_df.to_csv(self.CID_text_file_path, index=None)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def load_Graph_CID_and_text(self):


        self.graphs, self.slices,= torch.load(self.processed_paths[0])

        CID_text_df = pd.read_csv(self.CID_text_file_path)
        self.CID_list = CID_text_df["CID"].tolist()
        self.text_list = CID_text_df["text"].tolist()
        return

    def __len__(self):
        return len(self.graphs.text)

    def get(self,idx):
        text = self.text_list[idx]
        data = Data()
        data.text = text
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            if key == 'smiles' or key == 'text':
                data[key] = item[idx]
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return data

class PubChemSTM_SubDatasets_Motif(PubChemSTM_Datasets_Motif):
    def __init__(self, root, datafile, size, transform=None, pre_transform=None, pre_filter=None):
        self.size = size
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.datafile = "Map_" + datafile
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/{}/CID_text_list.csv".format(self.datafile[:-3]))
        if not os.path.exists(os.path.join(self.root, "processed/{}".format(self.datafile[:-3]))):
            os.makedirs(os.path.join(self.root, "processed/{}".format(self.datafile[:-3])))
        self.CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")

        self.tokenizer = Tokenizer("../data/PubChemSTM_data/vocab3.txt")
        self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}

        super(PubChemSTM_Datasets_Motif, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return
    def __len__(self):
        return self.size


class PubChemSTM_Datasets_Motif2(InMemoryDataset):
    def __init__(self, root, datafile, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.datafile = datafile
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/{}/CID_text_list.csv".format(self.datafile[:-3]))
        if not os.path.exists(os.path.join(self.root, "processed/{}".format(self.datafile[:-3]))):
            os.makedirs(os.path.join(self.root, "processed/{}".format(self.datafile[:-3])))
        self.CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")

        self.CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")

        self.tokenizer = Tokenizer("../data/PubChemSTM_data/vocab3.txt")
        self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}

        super(PubChemSTM_Datasets_Motif2, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        # return 'geometric_data_motif_processedtmp1006.pt'
        # return 'geometric_data_motif_mask11_processed.pt'
        print(self.datafile)
        return self.datafile

    def process(self):

        suppl = Chem.SDMolSupplier(self.SDF_file_path)
        CID2graph = {}
        valid = 0
        # count = 0

        for mol in tqdm(suppl):
            # count = count + 1
            # if count == 50:
            #     break
            try:
                CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
            except:
                continue
            CID = int(CID)
            graph = mol_to_graph_data_obj_simple(mol)
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
            graph.smiles = smiles
            try:
                tree = self.tokenizer(smiles)
            except:
                print("Unable to process SMILES:", smiles)
                continue

            # Manually consructing the fragment graph
            map = [0]*graph.num_nodes
            frag = [[0] for _ in range(len(tree.nodes))]
            frag_edge_index = [[],[]]

            try:
                for node_i in tree.nodes:
                    node = tree.get_node(node_i)
                    # for atom in node, set map
                    for atom_i in node.atom_mapping.keys():
                        map[atom_i] = node_i
                    # extend frag
                        frag[node_i][0] = self.vocab_dict[node.smiles]
                for src, dst in tree.edges:
                    # extend edge index
                    frag_edge_index[0].extend([src,dst])
                    frag_edge_index[1].extend([dst,src])
            except KeyError as e:
                print("Error in matching subgraphs", e)
                continue

            # unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
            # frag_unique = torch.zeros(800).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

            graph.clique = torch.LongTensor(map)
            graph.maskids = list(range(len(frag)))
            graph.motiflabel = [item for sublist in frag for item in sublist]
            # graph.frag_edge_index = torch.LongTensor(frag_edge_index)
            # graph.frag_unique = frag_unique
            CID2graph[CID] = graph


        print("CID2graph", len(CID2graph))
        print(valid)

        with open(self.CID2text_file, "r") as f:
            CID2text_data = json.load(f)
        print("CID2data", len(CID2text_data))

        CID_list, graph_list, text_list = [], [], []
        missing = 0
        for CID, value_list in CID2text_data.items():
            CID = int(CID)
            if CID not in CID2graph:
                print("CID {} missing".format(CID))
                missing = missing + 1
                continue
            graph = CID2graph[CID]

            # clique = CID2clique[CID]
            for value in value_list:
                text_list.append(value)
                CID_list.append(CID)
                graph.text = value
                graph_list.append(graph)

        print("total missing: {}".format(missing))
        CID_text_df = pd.DataFrame({"CID": CID_list, "text": text_list})
        CID_text_df.to_csv(self.CID_text_file_path, index=None)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def load_Graph_CID_and_text(self):


        self.graphs, self.slices,= torch.load(self.processed_paths[0])

        # CID_text_df = pd.read_csv(self.CID_text_file_path)
        # self.CID_list = CID_text_df["CID"].tolist()
        # self.text_list = CID_text_df["text"].tolist()
        return

    def __len__(self):
        return len(self.graphs.text)

    def get(self,idx):
        # text = self.text_list[idx]
        # clique = self.cliques[idx] # TODO

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            elif key == "text":
                text = item[idx]
                continue
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]

        return text, data  # clique


