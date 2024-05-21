import torch
from torch_geometric.data import Dataset

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

class RetrievalDataset(InMemoryDataset):
    def __init__(self, root, args):
        super(RetrievalDataset, self).__init__(root)
        self.root = root
        self.graph_aug = 'noaug'
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir('kv_data/test/graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir('kv_data/test/text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root + 'smiles/')
        self.smiles_name_list.sort()
        self.load_Graph_CID_and_text()
        # self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        # self.use_smiles = args.use_smiles

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)



    def __getitem__(self, index):
        graph_name, text_name, smiles_name = self.graph_name_list[index], self.text_name_list[index], self.smiles_name_list[index]
        assert graph_name[len('graph_'):-len('.pt')] == text_name[len('text_'):-len('.txt')] == smiles_name[len('smiles_'):-len('.txt')], print(graph_name, text_name, smiles_name)

        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        text = ''
        # if self.use_smiles:
        #     text_path = os.path.join(self.root, 'smiles', smiles_name)
        #     text = 'This molecule is '
        #     count = 0
        #     for line in open(text_path, 'r', encoding='utf-8'):
        #         count += 1
        #         line = line.strip('\n')
        #         text += f' {line}'
        #         if count > 1:
        #             break
        #     text += '. '
        
        text_path = os.path.join(self.root, 'text', text_name)
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text += f' {line}'
            if count > 100:
                break
        text += '\n'
        # para-level
        text, mask = self.tokenizer_text(text)
        return data_graph, text.squeeze(0), mask.squeeze(0)  # , index

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


class RetrievalDatasetKVPLM(InMemoryDataset):
    def __init__(self, root, args):
        super(RetrievalDatasetKVPLM, self).__init__(root)
        self.root = root
        self.graph_aug = 'noaug'
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        # self.smiles_name_list = os.listdir(root + 'smiles/')
        # self.smiles_name_list.sort()
        # self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        # self.use_smiles = args.use_smiles

    def processed_file_names(self):
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

        for mol in tqdm(suppl):
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
                results.append(smiles)

                # if idx >=7 and idx <= 2456 and random.randint(0, 9) < 2:
                #     maskids.append(num)
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
        with open("../data/PubChemSTM_data/motifs2.txt", 'a') as file:
            for item in results:
                file.write(item + '\n')

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
        # self.CID_list = CID_text_df["CID"].tolist()
        self.text_list = CID_text_df["text"].tolist()
        return

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
        # assert graph_name[len('graph_'):-len('.pt')] == text_name[len('text_'):-len('.txt')] == smiles_name[len('smiles_'):-len('.txt')], print(graph_name, text_name, smiles_name)

        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        text = ''
        text_path = os.path.join(self.root, 'text', text_name)
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line = line.strip('\n')
            text += f' {line}'
            break
        text += '\n'
        # para-level
        text, mask = self.tokenizer_text(text)
        return data_graph, text.squeeze(0), mask.squeeze(0)  # , index

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask
