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
from rdkit.Chem import BRICS
RDLogger.DisableLog('rdApp.*')

from FineMolTex.datasets.utils import mol_to_graph_data_obj_simple


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
        self.CID_text_file_path = os.path.join(self.root, "processed/CID_text_list.csv")

        super(PubChemSTM_Datasets_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

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

def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # strategy 1: break bonds between rings and non-ring atoms
    for c in cliques:
        if len(c) > 1:
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                cliques.remove(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                cliques.remove(c)
                cliques.append([c[0]])
                breaks.append(c)

    # strategy 2: select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 2 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles) # 创建一个 RDKit 分子对象（mol
    if mol is None:
        return None
    Chem.Kekulize(mol) # 确保生成的分子对象遵循 Kekulé 结构规则。Kekulization 是一种化学信息处理中的标准化过程，用于处理分子中的芳香环结构。
    return mol


def get_smiles(mol):
    if mol == None:
        return "*"
    try:
        a = Chem.MolToSmiles(mol, kekuleSmiles=True)
    except:
        a = Chem.MolToSmiles(mol, kekuleSmiles=False)
    return a

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def get_clique_mol(mol, atoms):
    # get the fragment of clique
    # 根据给定的原子索引列表 atoms 从给定的分子 mol 中提取一个特定的原子团（fragment）
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True) # 提取具有指定原子索引列表 atoms 的分子片段的 SMILES 表示
    except:
        print("Error")
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=False)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # 对 new_mol 分子执行一系列的操作，包括修正分子结构，以确保其有效性和正确性。这通常包括处理分子键的立体化和构象。
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    if new_mol == None:
        a = 1
    return new_mol

class PubChemSTM_Datasets_GraphMotif(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/CID_text_list.csv")
        self.CID2SMILES_file = os.path.join(self.root, "raw/CID2SMILES.csv")

        super(PubChemSTM_Datasets_GraphMotif, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    @property
    def processed_file_names(self):
        # return 'geometric_data_motif_processed.pt'
        return 'geometric_data_motif_mask_processed.pt'

    def process(self):
        df = pd.read_csv(self.CID2SMILES_file)
        results = []
        with open("../data/PubChemSTM_data/motifs.txt", 'r') as file:
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
            cliques2 = torch.zeros(graph.x.shape[0])
            num = 0
            labels = []
            maskids = []
            for clique in cliques:
                cmol = get_clique_mol(mol, clique)
                smiles = get_smiles(cmol)
                # smi = Chem.MolFragmentToSmiles(mol, clique, kekuleSmiles=False)
                idx = results.index(smiles)

                if idx >=7 and idx <= 11042 and random.randint(0, 9) >=5:
                    maskids.append(num)

                labels.append(idx)
                cliques2[clique] = num
                num = num + 1


                # for clique in cliques:

            graph.clique = cliques2.to(torch.int64)
            graph.motiflabel = labels
            graph.maskids = maskids
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
        # clique = self.cliques[idx] # TODO

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            if key == 'smiles' or key == 'maskids':
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]

        return text, data, # clique

    def __len__(self):
        return len(self.text_list)


class PubChemSTM_SubDatasets_GraphMotif(PubChemSTM_Datasets_GraphMotif):
    def __init__(self, root, size, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.size = size
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.size = size
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/CID_text_list.csv")

        super(PubChemSTM_Datasets_GraphMotif, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    def __len__(self):
        return self.size

class PubChemSTM_SubDatasets_Graph(PubChemSTM_Datasets_Graph):
    def __init__(self, root, size, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.size = size
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.size = size
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/CID_text_list.csv")
        
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
