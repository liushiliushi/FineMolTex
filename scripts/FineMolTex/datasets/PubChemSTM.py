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

from MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple, get_positions, tree_decomp, brics_decomp, get_clique_mol, get_smiles


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


class PubChemSTM_Datasets_SMILES(Dataset):
    def __init__(self, root):
        self.root = root

        CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
        self.load_CID2SMILES(CID2text_file, CID2SMILES_file)
        
        self.text_list = []
        missing_count = 0
        for CID, value_list in self.CID2text_data.items():
            if CID not in self.CID2SMILES:
                print("CID {} missing".format(CID))
                missing_count += 1
                continue
            for value in value_list:
                self.text_list.append([CID, value])
        print("missing", missing_count)
        print("len of text_list: {}".format(len(self.text_list)))
        return
    
    def load_CID2SMILES(self, CID2text_file, CID2SMILES_file):
        with open(CID2text_file, "r") as f:
            self.CID2text_data = json.load(f)
        print("len of CID2text: {}".format(len(self.CID2text_data.keys())))

        df = pd.read_csv(CID2SMILES_file)
        CID_list, SMILES_list = df["CID"].tolist(), df["SMILES"].tolist()
        self.CID2SMILES = {}
        for CID, SMILES in zip(CID_list, SMILES_list):
            CID = str(CID)
            self.CID2SMILES[CID] = SMILES
        print("len of CID2SMILES: {}".format(len(self.CID2SMILES.keys())))
        return

    def __getitem__(self, index):
        CID, text = self.text_list[index]
        SMILES = self.CID2SMILES[CID]
        return text, SMILES

    def __len__(self):
        return len(self.text_list)


class PubChemSTM_SubDatasets_SMILES(PubChemSTM_Datasets_SMILES):
    def __init__(self, root, size):
        self.root = root

        CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")
        self.load_CID2SMILES(CID2text_file, CID2SMILES_file)
        
        self.text_list = []
        for CID, value_list in self.CID2text_data.items():
            if CID not in self.CID2SMILES:
                print("CID {} missing".format(CID))
                continue
            for value in value_list:
                self.text_list.append([CID, value])
            if len(self.text_list) >= size:
                break
        print("len of text_list: {}".format(len(self.text_list)))
        return

class PubChemSTM_Datasets_Graph(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/STM/CID_text_list.csv")
        if not os.path.exists(os.path.join(self.root, "processed/STM")):
            os.makedirs(os.path.join(self.root, "processed/STM"))


        super(PubChemSTM_Datasets_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processedtmp.pt'

    def process(self):
        suppl = Chem.SDMolSupplier(self.SDF_file_path)

        CID2graph = {}
        for mol in tqdm(suppl):
            try:
                CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
            except:
                continue
            CID = int(CID)
            graph = mol_to_graph_data_obj_simple(mol)
            CID2graph[CID] = graph
        print("CID2graph", len(CID2graph))

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
            for value in value_list:
                text_list.append(value)
                CID_list.append(CID)
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
        self.graphs, self.slices = torch.load(self.processed_paths[0])

        CID_text_df = pd.read_csv(self.CID_text_file_path)
        self.CID_list = CID_text_df["CID"].tolist()
        self.text_list = CID_text_df["text"].tolist()
        return

    def get(self, idx):
        text = self.text_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            if key == "smiles":
                data[key] = item[idx]
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return text, data

    def __len__(self):
        return len(self.text_list)


class PubChemSTM_Datasets_GraphMotif(InMemoryDataset):
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

        super(PubChemSTM_Datasets_GraphMotif, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        # return 'geometric_data_motif_processedtmp125.pt'
        # return 'geometric_data_motif_mask10_processed.pt'
        print(self.datafile)
        return self.datafile

    def process(self):
        # df = pd.read_csv(self.CID2SMILES_file)
        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)

        suppl = Chem.SDMolSupplier(self.SDF_file_path)
        CID2graph = {}
        valid = 0
        count = 0
        for mol in tqdm(suppl):
            count = count + 1
            if count == 1000:
                break
            try:
                CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
            except:
                continue
            CID = int(CID)
            graph = mol_to_graph_data_obj_simple(mol)
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
            graph.smiles = smiles
            cliques, edges = brics_decomp(mol)
            if len(edges) <= 1:
                cliques, edges = tree_decomp(mol)
                if len(cliques) == 0:
                    cliques, edges = brics_decomp(mol)
            cliques2 = torch.zeros(graph.x.shape[0])
            positions = get_positions(cliques, edges)
            # if len(positions) != len(cliques):
            #     stop = 1
            num = 0
            labels = []
            maskids = []
            for clique in cliques:
                cmol = get_clique_mol(mol, clique)
                smiles = get_smiles(cmol)

                idx = results.index(smiles)
                # results.append(smiles)

                if idx >=0 and idx <= 2456:
                    maskids.append(num)
                    # print("num:{} label:{}".format(num, idx-7))
                # labels.append(idx-7)
                labels.append(idx)

                cliques2[clique] = num
                num = num + 1


                # for clique in cliques:
            graph.positions = positions
            graph.clique = cliques2.to(torch.int64)
            graph.motiflabel = labels
            graph.maskids = maskids
            CID2graph[CID] = graph

        # results = set(results)
        # with open("../data/PubChemSTM_data/motifs2.txt", 'a') as file:
        #     for item in results:
        #         file.write(item + '\n')

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

    def get(self, idx):
        text = self.text_list[idx]
        cid = self.CID_list[idx]
        # clique = self.cliques[idx] # TODO

        data = Data()
        for key in self.graphs.keys():
            item, slices = self.graphs[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            elif key == "text":
                continue
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        data.cid = cid
        return text, data # clique

    def __len__(self):
        return len(self.text_list)



class PubChemSTM_SubDatasets_GraphMotif(PubChemSTM_Datasets_GraphMotif):
    def __init__(self, root, datafile, size, transform=None, pre_transform=None, pre_filter=None):
        self.size = size
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

        super(PubChemSTM_Datasets_GraphMotif, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    def __len__(self):
        return self.size

class PubChemSTM_SubDatasets_Graph(PubChemSTM_Datasets_Graph):
    def __init__(self, root, size, transform=None, pre_transform=None, pre_filter=None):
        # self.root = root
        # self.size = size
        # self.transform = transform
        # self.pre_transform = pre_transform
        # self.pre_filter = pre_filter
        # self.size = size
        # # only for `process` function
        # self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        # self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # # `process` result file
        # self.CID_text_file_path = os.path.join(self.root, "processed/CID_text_list.csv")
        self.size = size

        super(PubChemSTM_Datasets_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    def __len__(self):
        return self.size


class PubChemSTM_Datasets_SMILES_and_Graph(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        # `process` result file
        self.SMILES_file_path = os.path.join(self.root, "processed_molecule_only/SMILES.csv")
        
        super(PubChemSTM_Datasets_SMILES_and_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

        CID_text_df = pd.read_csv(self.SMILES_file_path)
        self.SMILES_list = CID_text_df["smiles"].tolist()
        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_molecule_only')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        suppl = Chem.SDMolSupplier(self.SDF_file_path)

        SMILES_list, graph_list = [], []
        for mol in tqdm(suppl):
            SMILES = Chem.MolToSmiles(mol)
            SMILES_list.append(SMILES)
            graph = mol_to_graph_data_obj_simple(mol) # rdkit mol --> pyg graph
            graph_list.append(graph)

        SMILES_df = pd.DataFrame({"smiles": SMILES_list})
        SMILES_df.to_csv(self.SMILES_file_path, index=None)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list) # 将图列表转换为一个大的批处理数据对象和切片索引。
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def get(self, idx):
        SMILES = self.SMILES_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return SMILES, data

    def __len__(self):
        return len(self.SMILES_list)
