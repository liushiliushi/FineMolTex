import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data
import os
import random
from itertools import repeat
from FineMolTex.datasets.utils import mol_to_graph_data_obj_simple, get_positions, tree_decomp, brics_decomp, get_clique_mol, get_smiles
from rdkit import Chem


class PubChemDataset(InMemoryDataset):
    def __init__(self, root, datafile):

        self.root = root
        self.datafile = datafile
        self.tokenizer = None
        self.datafile = datafile

        super(PubChemDataset, self).__init__(root)
        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        return 'geometric_data2_motif_processed5.pt'
        # return 'geometric_data_motif_mask10_processed.pt'
        print(self.datafile)
        return self.datafile

    def load_Graph_CID_and_text(self):
        self.graphs, self.slices,= torch.load(self.processed_paths[0])

        return

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graphs.text)

    def __getitem__(self, idx):
        graph = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            if key == 'x' or key == 'edge_index' or key == "edge_attr" or key == "clique":
                s = list(repeat(slice(None), item.dim()))
                s[graph.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                graph[key] = item[s]
            else:
                graph[key] = item[idx]


        text = graph.text
        return text, graph

    def process(self):
        # df = pd.read_csv(self.CID2SMILES_file)
        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)
        data, slices2 = torch.load("{}/pretrain.pt".format(self.root))
        graph_list = []
        for idx in range(len(data.text)):
        # for idx in range(100):
            graph = Data()
            for key in data.keys:
                item, slices = data[key], slices2[key]
                if key == 'x' or key == 'edge_index' or key == "edge_attr":
                    s = list(repeat(slice(None), item.dim()))
                    s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                    graph[key] = item[s]
                else:
                    graph[key] = item[idx]
            mol = Chem.MolFromSmiles(graph.smiles)
            cliques, edges = brics_decomp(mol)
            if len(edges) <= 1:
                cliques, edges = tree_decomp(mol)
                if len(cliques) == 0:
                    cliques, edges = brics_decomp(mol)
            cliques2 = torch.zeros(graph.x.shape[0])
            positions = get_positions(cliques, edges)
            num = 0
            labels = []
            maskids = []
            for clique in cliques:
                cmol = get_clique_mol(mol, clique)
                smiles = get_smiles(cmol)
                try:
                    idx = results.index(smiles)
                except:
                    idx = 0
                results.append(smiles)

                if idx >= 7 and idx <= 106 and random.randint(0, 9) < 1:
                    maskids.append(num)
                    # print("num:{} label:{}".format(num, idx-7))
                labels.append(idx - 7)

                cliques2[clique] = num
                num = num + 1

                # for clique in cliques:
            graph.positions = positions
            graph.clique = cliques2.to(torch.int64)
            graph.motiflabel = labels
            graph.maskids = maskids
            graph_list.append(graph)

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])
        return




