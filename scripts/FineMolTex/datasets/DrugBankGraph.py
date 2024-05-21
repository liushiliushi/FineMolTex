import os
from itertools import chain, repeat
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple, get_positions, tree_decomp, brics_decomp, get_clique_mol, get_smiles
from rdkit.Chem import AllChem
from MoleculeSTM.datasets.mol_bpe import Tokenizer
import random


class DrugBank_Datasets_Graph_retrieval(InMemoryDataset):
    def __init__(
        self, root, train_mode, neg_sample_size, processed_dir_prefix, template="raw/SMILES_description_{}.txt",
        transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.processed_dir_prefix = processed_dir_prefix
        self.template = template
        self.train_mode = train_mode
        self.smiles_text_file_name = "SMILES1.csv"

        super(DrugBank_Datasets_Graph_retrieval, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        print(df.columns)
        self.text_list = df["text"].tolist()

        # sampling
        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", template.format(train_mode))
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list
        
        return

    def get_graph(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return data

    def get(self, index):
        text = self.text_list[index]
        data = self.get_graph(index)
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.text_list[idx] for idx in neg_index_list]
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get_graph(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', '{}_{}'.format(self.processed_dir_prefix, self.train_mode))

    @property
    def processed_file_names(self):
        return 'geometric_data_drug_bank_processed8.pt'

    def download(self):
        return

    def process(self):
        data_list, SMILES_list, text_list = [], [], []
        SMILES2description_file = os.path.join(self.root, 'raw', self.template.format(self.train_mode))
        f = open(SMILES2description_file, 'r')

        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)

        for line_id, line in enumerate(f.readlines()):
            line = line.strip().split("\t", 1)
            SMILES = line[0]
            text = line[1]

            mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(mol)
            data.smiles = SMILES
            data.id = torch.tensor([line_id])
            cliques, edges = brics_decomp(mol)
            if len(edges) <= 1:
                cliques, edges = tree_decomp(mol)
                if len(cliques) == 0:
                    cliques, edges = brics_decomp(mol)
            cliques2 = torch.zeros(data.x.shape[0])
            positions = get_positions(cliques, edges)

            num = 0
            labels2 = []
            maskids = []
            for clique in cliques:
                cmol = get_clique_mol(mol, clique)
                smiles = get_smiles(cmol)
                if smiles in results:
                    idx = results.index(smiles)
                else:
                    idx = 20000
                if idx >=0 and idx <= 2456:
                    maskids.append(num)
                # results.append(smiles)

                labels2.append(idx)
                cliques2[clique] = num
                num = num + 1

            data.positions = positions
            data.clique = cliques2.to(torch.int64)
            data.motiflabel = labels2
            data.maskids = maskids
            data_list.append(data)
            SMILES_list.append(SMILES)
            text_list.append(text)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"text": text_list, "smiles": SMILES_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        print()
        return

    def __len__(self):
        return len(self.text_list)


class DrugBank_Datasets_Motif_retrieval(InMemoryDataset):
    def __init__(
            self, root, train_mode, neg_sample_size, processed_dir_prefix, template="raw/SMILES_description_{}.txt",
            transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.processed_dir_prefix = processed_dir_prefix
        self.template = template
        self.train_mode = train_mode
        self.smiles_text_file_name = "SMILES.csv"
        self.tokenizer = Tokenizer("../data/PubChemSTM_data/vocab3.txt")
        self.vocab_dict = {smiles: i for i, smiles in enumerate(self.tokenizer.vocab_dict.keys())}

        super(DrugBank_Datasets_Motif_retrieval, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.graphs, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        print(df.columns)
        self.text_list = df["text"].tolist()

        # sampling
        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", template.format(train_mode))
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list

        return


    def getplain(self, index):
        if index >= len(self.text_list):
            index = index - 100
        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            if key == 'smiles' or key == 'text':
                data[key] = item[index]
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[index], slices[index + 1])
                data[key] = item[s]
        return data
    def get(self, index):
        data = self.getplain(index)
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        data.neg_text = [self.getplain(idx).text for idx in neg_index_list]
        data.neg_data = [self.getplain(idx) for idx in neg_index_list]

        return data

    def getdata(self, index):
        data = self.get(index)
        text = data.text

        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.get(idx).text for idx in neg_index_list]
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', '{}_{}'.format(self.processed_dir_prefix, self.train_mode))

    @property
    def processed_file_names(self):
        return 'geometric_data_drug_bank_processedMotif6.pt'

    def download(self):
        return

    def process(self):
        data_list, SMILES_list, text_list = [], [], []
        SMILES2description_file = os.path.join(self.root, 'raw', self.template.format(self.train_mode))
        f = open(SMILES2description_file, 'r')

        results = []

        for line_id, line in enumerate(f.readlines()):
            line = line.strip().split("\t", 1)
            smiles = line[0]
            text = line[1]

            mol = AllChem.MolFromSmiles(smiles)
            graph = mol_to_graph_data_obj_simple(mol)
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
            graph.text = text

            data_list.append(graph)
            SMILES_list.append(smiles)
            text_list.append(text)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"text": text_list, "smiles": SMILES_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        print()
        return

    def __len__(self):
        return len(self.text_list)


class DrugBank_Datasets_Graph_retrieval_Sub(InMemoryDataset):
    def __init__(
            self, root, train_mode, neg_sample_size, processed_dir_prefix, template="raw/SMILES_description_{}.txt",
            transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.processed_dir_prefix = processed_dir_prefix
        self.template = template
        self.train_mode = train_mode
        self.smiles_text_file_name = "SMILES.csv"

        super(DrugBank_Datasets_Graph_retrieval_Sub, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        print(df.columns)
        self.text_list = df["text"].tolist()

        # sampling
        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", template.format(train_mode))
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list

        return

    def get_graph(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return data

    def get(self, index):
        text = self.text_list[index]
        data = self.get_graph(index)
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.text_list[idx] for idx in neg_index_list]
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get_graph(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', '{}_{}'.format(self.processed_dir_prefix, self.train_mode))

    @property
    def processed_file_names(self):
        return 'geometric_data_drug_bank_processed.pt'

    def download(self):
        return

    def process(self):
        data_list, SMILES_list, text_list = [], [], []
        SMILES2description_file = os.path.join(self.root, 'raw', self.template.format(self.train_mode))
        f = open(SMILES2description_file, 'r')

        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)

        for line_id, line in enumerate(f.readlines()):
            line = line.strip().split("\t", 1)
            SMILES = line[0]
            text = line[1]

            mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(mol)
            data.smiles = SMILES
            data.id = torch.tensor([line_id])
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

                labels.append(idx - 7)
                cliques2[clique] = num
                num = num + 1

            data.positions = positions
            data.clique = cliques2.to(torch.int64)
            data_list.append(data)
            SMILES_list.append(SMILES)
            text_list.append(text)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"text": text_list, "smiles": SMILES_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        print()
        return

    def __len__(self):
        return 10


class DrugBank_Datasets_Graph_ATC(InMemoryDataset):
    def __init__(
        self, root, file_name, processed_dir_prefix, neg_sample_size, prompt_template="{}.",
        transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.file_name = file_name
        self.processed_dir_prefix = processed_dir_prefix
        self.smiles_text_file_name = "SMILES.csv"
        self.prompt_template = prompt_template

        super(DrugBank_Datasets_Graph_ATC, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        self.SMILES_list = df["smiles"].tolist()
        self.ATC_code_list = df["ATC_code"].tolist()
        ATC_label_list = df["ATC_label"].tolist() # This is for raw TAC label
        self.ATC_label_list = [self.prompt_template.format(x) for x in ATC_label_list]

        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", file_name)
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list

        assert len(self.SMILES_list) == len(self.neg_index_list) == len(self.ATC_code_list) == len(self.ATC_label_list)
        return

    def get_graph(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return data

    def get(self, index):
        text = self.ATC_label_list[index]
        data = self.get_graph(index)
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.ATC_label_list[idx] for idx in neg_index_list]
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get_graph(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", "molecule_{}".format(self.processed_dir_prefix))

    @property
    def processed_file_names(self):
        return 'geometric_data_atc_2457_processed2.pt'

    def download(self):
        return

    def process(self):
        SMILES2ATC_txt_file = os.path.join(self.root, "raw", self.file_name)
        
        f = open(SMILES2ATC_txt_file, 'r')
        data_list, SMILES_list, ATC_code_list, ATC_label_list = [], [], [], []
        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)

        for line_idx, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            SMILES = line[0]
            ATC_code = line[1]
            ATC_label = line[2]
            mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(mol)
            data.id = torch.tensor([line_idx])

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
            maskids = []
            for clique in cliques:
                cmol = get_clique_mol(mol, clique)
                smiles = get_smiles(cmol)
                if smiles in results:
                    idx = results.index(smiles)
                else:
                    idx = 20000
                if idx >=0 and idx <= 2456:
                    maskids.append(num)
                    # print("num:{} label:{}".format(num, idx-7))
                # labels.append(idx-7)
                labels.append(idx)
                results.append(smiles)
                cliques2[clique] = num
                num = num + 1

            data.positions = positions
            data.clique = cliques2.to(torch.int64)
            data.motiflabel = labels
            data.maskids = maskids
            data_list.append(data)
            SMILES_list.append(SMILES)
            ATC_code_list.append(ATC_code)
            ATC_label_list.append(ATC_label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"smiles": SMILES_list, "ATC_code": ATC_code_list, "ATC_label": ATC_label_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        return

    def __len__(self):
        return len(self.SMILES_list)


class DrugBank_Datasets_Graph_ATC_rand(InMemoryDataset):
    def __init__(
            self, root, file_name, processed_dir_prefix, neg_sample_size, prompt_template="{}.",
            transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.file_name = file_name
        self.processed_dir_prefix = processed_dir_prefix
        self.smiles_text_file_name = "SMILES.csv"
        self.prompt_template = prompt_template

        super(DrugBank_Datasets_Graph_ATC_rand, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        self.SMILES_list = df["smiles"].tolist()
        self.ATC_code_list = df["ATC_code"].tolist()
        ATC_label_list = df["ATC_label"].tolist()  # This is for raw TAC label
        self.ATC_label_list = [self.prompt_template.format(x) for x in ATC_label_list]

        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", file_name)
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for i in range(len(self.ATC_label_list)):
            line = []
            while len(line) < 50:
                num = random.randint(0, (len(self.ATC_label_list) - 1))
                if num != i:
                    line.append(num)
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list

        assert len(self.SMILES_list) == len(self.neg_index_list) == len(self.ATC_code_list) == len(self.ATC_label_list)
        return

    def get_graph(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return data

    def get(self, index):
        text = self.ATC_label_list[index]
        data = self.get_graph(index)
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.ATC_label_list[idx] for idx in neg_index_list]
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get_graph(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", "molecule_{}".format(self.processed_dir_prefix))

    @property
    def processed_file_names(self):
        return 'geometric_data_atc_processed.pt'

    def download(self):
        return

    def process(self):
        SMILES2ATC_txt_file = os.path.join(self.root, "raw", self.file_name)

        f = open(SMILES2ATC_txt_file, 'r')
        data_list, SMILES_list, ATC_code_list, ATC_label_list = [], [], [], []
        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)

        for line_idx, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            SMILES = line[0]
            ATC_code = line[1]
            ATC_label = line[2]
            mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(mol)
            data.id = torch.tensor([line_idx])

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

                labels.append(idx - 7)
                cliques2[clique] = num
                num = num + 1

            data.positions = positions
            data.clique = cliques2.to(torch.int64)
            data_list.append(data)
            SMILES_list.append(SMILES)
            ATC_code_list.append(ATC_code)
            ATC_label_list.append(ATC_label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"smiles": SMILES_list, "ATC_code": ATC_code_list, "ATC_label": ATC_label_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        return

    def __len__(self):
        return len(self.SMILES_list)


class DrugBank_Datasets_Graph_ATC_Sub(InMemoryDataset):
    def __init__(
            self, root, file_name, processed_dir_prefix, neg_sample_size, prompt_template="{}.",
            transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.file_name = file_name
        self.processed_dir_prefix = processed_dir_prefix
        self.smiles_text_file_name = "SMILES.csv"
        self.prompt_template = prompt_template

        super(DrugBank_Datasets_Graph_ATC_Sub, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        self.SMILES_list = df["smiles"].tolist()
        self.ATC_code_list = df["ATC_code"].tolist()
        ATC_label_list = df["ATC_label"].tolist()  # This is for raw TAC label
        self.ATC_label_list = [self.prompt_template.format(x) for x in ATC_label_list]

        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", file_name)
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list

        assert len(self.SMILES_list) == len(self.neg_index_list) == len(self.ATC_code_list) == len(self.ATC_label_list)
        return

    def get_graph(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if key == 'smiles' or key == 'maskids' or key == "positions":
                data[key] = item[idx]
            elif key == "motiflabel":
                data[key] = item[idx]
                # idx = random.sample(range(ptr[i], ptr[i + 1]), k=int(size * self.mask_ratio))
            else:
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
        return data

    def get(self, index):
        text = self.ATC_label_list[index]
        data = self.get_graph(index)
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.ATC_label_list[idx] for idx in neg_index_list]
        neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get_graph(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", "molecule_{}".format(self.processed_dir_prefix))

    @property
    def processed_file_names(self):
        return 'geometric_data_atc_processed.pt'

    def download(self):
        return

    def process(self):
        SMILES2ATC_txt_file = os.path.join(self.root, "raw", self.file_name)

        f = open(SMILES2ATC_txt_file, 'r')
        data_list, SMILES_list, ATC_code_list, ATC_label_list = [], [], [], []
        results = []
        with open("../data/PubChemSTM_data/vocab2.txt", 'r') as file:
            for line in file:
                line = line.strip('\n')
                results.append(line)

        for line_idx, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            SMILES = line[0]
            ATC_code = line[1]
            ATC_label = line[2]
            mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(mol)
            data.id = torch.tensor([line_idx])

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

                labels.append(idx - 7)
                cliques2[clique] = num
                num = num + 1

            data.positions = positions
            data.clique = cliques2.to(torch.int64)
            data_list.append(data)
            SMILES_list.append(SMILES)
            ATC_code_list.append(ATC_code)
            ATC_label_list.append(ATC_label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"smiles": SMILES_list, "ATC_code": ATC_code_list, "ATC_label": ATC_label_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        return

    def __len__(self):
        return 10

