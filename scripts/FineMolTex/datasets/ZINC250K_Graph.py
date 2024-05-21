import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from itertools import repeat

import torch
from torch_geometric.data import Data, InMemoryDataset

from MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple, get_positions, tree_decomp, brics_decomp, get_clique_mol, get_smiles



class ZINC250K_Dataset_GraphMotif(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        self.SMILES_file = os.path.join(self.root, "raw/250k_rndm_zinc_drugs_clean_3.csv")
        df = pd.read_csv(self.SMILES_file)
        SMILES_list = df['smiles'].tolist()
        self.SMILES_list = [x.strip() for x in SMILES_list]

        super(ZINC250K_Dataset_GraphMotif, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_molecule_only')

    @property
    def processed_file_names(self):
        # print(self.datafile)
        # return self.datafile
        return 'geometric_data_motif_processed.pt'

    def process(self):
        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)
        graph_list = []
        for SMILES in tqdm(self.SMILES_list):
            mol = Chem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(mol)
            data.smiles = SMILES
            cliques, edges = brics_decomp(mol)
            if len(edges) <= 1:
                cliques, edges = tree_decomp(mol)
                if len(cliques) == 0:
                    cliques, edges = brics_decomp(mol)
            cliques2 = torch.zeros(data.x.shape[0])
            positions = get_positions(cliques, edges)

            num = 0
            labels = []
            for clique in cliques:
                cmol = get_clique_mol(mol, clique)
                smiles = get_smiles(cmol)
                if smiles in results:
                    idx = results.index(smiles)
                else:
                    idx = 20000
                results.append(smiles)

                labels.append(idx)
                cliques2[clique] = num
                num = num + 1

            data.positions = positions
            data.clique = cliques2.to(torch.int64)
            data.motiflabel = labels

            graph_list.append(data)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def get(self, idx):
        SMILES = self.SMILES_list[idx]

        data = Data()
        for key in self.graphs.keys():
            item, slices = self.graphs[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return SMILES, data

    def __len__(self):
        return len(self.SMILES_list)

class ZINC250K_Dataset_GraphMotifSub(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        self.SMILES_file = os.path.join(self.root, "raw/250k_rndm_zinc_drugs_clean_3.csv")
        df = pd.read_csv(self.SMILES_file)
        SMILES_list = df['smiles'].tolist()
        self.SMILES_list = [x.strip() for x in SMILES_list]

        super(ZINC250K_Dataset_GraphMotifSub, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

        if subset_size is not None:
            self.SMILES_list = self.SMILES_list[:subset_size]
        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed_molecule_only')

    @property
    def processed_file_names(self):
        # print(self.datafile)
        # return self.datafile
        return 'geometric_data_motif_processed.pt'

    def process(self):
        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)
        graph_list = []
        for SMILES in tqdm(self.SMILES_list):
            mol = Chem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(mol)
            data.smiles = SMILES
            cliques, edges = brics_decomp(mol)
            if len(edges) <= 1:
                cliques, edges = tree_decomp(mol)
                if len(cliques) == 0:
                    cliques, edges = brics_decomp(mol)
            cliques2 = torch.zeros(data.x.shape[0])
            positions = get_positions(cliques, edges)

            num = 0
            labels = []
            for clique in cliques:
                cmol = get_clique_mol(mol, clique)
                smiles = get_smiles(cmol)
                if smiles in results:
                    idx = results.index(smiles)
                else:
                    idx = 20000
                results.append(smiles)

                labels.append(idx)
                cliques2[clique] = num
                num = num + 1

            data.positions = positions
            data.clique = cliques2.to(torch.int64)
            data.motiflabel = labels

            graph_list.append(data)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return

    def get(self, idx):
        SMILES = self.SMILES_list[idx]

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return SMILES, data

    def __len__(self):
        return 100

class ZINC250K_Dataset_Graph(InMemoryDataset):
    def __init__(self, root, subset_size=None, transform=None, pre_transform=None, pre_filter=None):
        self.root = root

        self.SMILES_file = os.path.join(self.root, "raw/250k_rndm_zinc_drugs_clean_3.csv")
        df = pd.read_csv(self.SMILES_file)
        SMILES_list = df['smiles'].tolist()
        self.SMILES_list = [x.strip() for x in SMILES_list]
        
        super(ZINC250K_Dataset_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.graphs, self.slices = torch.load(self.processed_paths[0])

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
        graph_list = []
        for SMILES in tqdm(self.SMILES_list):
            RDKit_mol = Chem.MolFromSmiles(SMILES)
            graph = mol_to_graph_data_obj_simple(RDKit_mol)
            graph_list.append(graph)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
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