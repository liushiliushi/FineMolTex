import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

from transformers import AutoModel, AutoTokenizer
from FineMolTex.datasets import DrugBank_Datasets_SMILES_retrieval, DrugBank_Datasets_Graph_retrieval, DrugBank_Datasets_Graph_retrieval_Sub
# from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
from FineMolTex.models import GNN, GNN_graphpred, GNN_motifpred, BertConnectionLayer, BertPreTrainingHeads, BertForMultiModalPreTraining,  BertConfig
from FineMolTex.utils import prepare_text_tokens, get_molecule_repr_MoleculeSTM, get_molecule_repr_MoleculeSTM2, freeze_network
import sys
result_dir = "F0206"
def do_CL_eval(X, Y, neg_Y, args):
    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1)  # B, 1, d

    Y = Y.unsqueeze(0)
    Y = torch.cat([Y, neg_Y], dim=0)  # T, B, d
    Y = Y.transpose(0, 1)  # B, T, d
    num_columns = Y.size(1)
    shuffled_indices = torch.randperm(num_columns)
    Y = Y[:, shuffled_indices, :]
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze()  # B*T
    B = X.size()[0]
    true_idx = torch.where(shuffled_indices==0)[0].item()
    labels = torch.zeros(B).long().to(logits.device)
    labels.fill_(true_idx)

    criterion = nn.CrossEntropyLoss()

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    confidence = logits
    CL_conf = confidence.max(dim=1)[0]
    CL_conf = CL_conf.cpu().numpy()

    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    return CL_loss, CL_conf, CL_acc

def get_text_repr(text):
    description_tokens_ids, description_masks = prepare_text_tokens(
        device=device, description=text, tokenizer=text_tokenizer, max_seq_len=args.max_seq_len)
    d_attention_mask = description_masks
    m_repr = torch.zeros((len(text), 512, config.v_hidden_size)).to(args.device)
    m_attention_mask = torch.full((len(text), 512), float(0)).to(args.device)
    m_position_embedding = torch.zeros((len(text), 512, 1)).to(args.device)

    prediction_scores_t, prediction_scores_m, all_attention_mask, pooled_output_t, pooled_output_m, _,_ = model(
        description_tokens_ids,
        m_repr,
        d_attention_mask,
        m_attention_mask,
        m_position_embedding=m_position_embedding,
        masked_lm_labels=None,
        image_label=None,
        image_target=None,
        next_sentence_label=None,
    )
    return pooled_output_t


@torch.no_grad()
def eval_epoch(dataloader):
    molecule_model.eval()
    mol2latent.eval()

    accum_acc_list_t = [0 for _ in args.T_list]
    accum_acc_list_m = [0 for _ in args.T_list]
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    for batch in L:
        print("Des")
        text = batch[0]
        molecule_data = batch[1]
        neg_text = batch[2]
        neg_molecule_data = batch[3]

        text_repr = get_text_repr(text)
        if args.molecule_type == "SMILES":
            molecule_data = list(molecule_data)  # for SMILES_list
            molecule_repr = get_molecule_repr_MoleculeSTM2(
                molecule_data, mol2latent=mol2latent,
                molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper, args=args)
        elif args.molecule_type == "Motif":
            molecule_data = molecule_data.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM2(
                molecule_data, mol2latent=mol2latent,
                molecule_type="Motif", molecule_model=molecule_model, model = model, device=args.device, args=args)


        else:
            molecule_data = molecule_data.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM2(
                molecule_data, mol2latent=mol2latent,
                molecule_type="Graph", molecule_model=molecule_model, args=args
            )

        if True:
            if args.molecule_type == "SMILES":
                neg_molecule_repr = [
                    get_molecule_repr_MoleculeSTM2(
                        list(neg_molecule_data[idx]), mol2latent=mol2latent,args=args,
                        molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper) for idx in range(T_max)
                ]
                neg_molecule_repr = torch.stack(neg_molecule_repr)
            else:
                neg_molecule_repr = [
                    get_molecule_repr_MoleculeSTM2(
                        neg_molecule_data[idx].to(device), mol2latent=mol2latent, args=args,  model = model,
                        molecule_type="Motif", molecule_model=molecule_model) for idx in range(T_max)
                ]
                neg_molecule_repr = torch.stack(neg_molecule_repr)
            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(text_repr, molecule_repr, neg_molecule_repr[:T - 1], args)
                accum_acc_list_t[T_idx] += acc

        if True:
            neg_text_repr = [get_text_repr(neg_text[idx]) for idx in range(T_max)]
            neg_text_repr = torch.stack(neg_text_repr)
            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(molecule_repr, text_repr, neg_text_repr[:T - 1], args)
                accum_acc_list_m[T_idx] += acc
        else:
            raise Exception

    accum_acc_list_t = np.array(accum_acc_list_t)
    accum_acc_list_t /= len(dataloader)

    accum_acc_list_m = np.array(accum_acc_list_m)
    accum_acc_list_m /= len(dataloader)
    return accum_acc_list_t, accum_acc_list_m


def eval_recall(dataloader):
    molecule_model.eval()
    mol2latent.eval()

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    for batch in L:
        # print("Des")
        text = batch[0]
        molecule_data = batch[1]

        text_repr = get_text_repr(text)
        if args.molecule_type == "SMILES":
            molecule_data = list(molecule_data)  # for SMILES_list
            molecule_repr = get_molecule_repr_MoleculeSTM2(
                molecule_data, mol2latent=mol2latent,
                molecule_type="SMILES", MegaMolBART_wrapper=MegaMolBART_wrapper, args=args)
        elif args.molecule_type == "Motif":
            molecule_data = molecule_data.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM2(
                molecule_data, mol2latent=mol2latent,
                molecule_type="Motif", molecule_model=molecule_model, model = model, device=args.device, args=args)


        else:
            molecule_data = molecule_data.to(device)
            molecule_repr = get_molecule_repr_MoleculeSTM2(
                molecule_data, mol2latent=mol2latent,
                molecule_type="Graph", molecule_model=molecule_model, args=args
            )


    accum_acc_list_t = np.array(accum_acc_list_t)
    accum_acc_list_t /= len(dataloader)

    accum_acc_list_m = np.array(accum_acc_list_m)
    accum_acc_list_m /= len(dataloader)
    return accum_acc_list_t, accum_acc_list_m


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--SSL_emb_dim", type=int, default=768)
    parser.add_argument("--text_type", type=str, default="SciBERT", choices=["SciBERT", "BioBERT"])
    parser.add_argument("--load_latent_projector", type=int, default=1)
    parser.add_argument("--model_loading_mode", type=str, default="load_from_latest", choices=["load_from_latest", "load_mode_0", "load_mode_1"])
    parser.add_argument("--training_mode", type=str, default="zero_shot", choices=["zero_shot"])

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--task", type=str, default="molecule_description",
        choices=[
            "molecule_description", "molecule_description_Raw",
            "molecule_description_removed_PubChem", "molecule_description_removed_PubChem_Raw",
            "molecule_pharmacodynamics", "molecule_pharmacodynamics_Raw",
            "molecule_pharmacodynamics_removed_PubChem", "molecule_pharmacodynamics_removed_PubChem_Raw"])
    parser.add_argument("--test_mode", type=str, default="given_text", choices=["given_text", "given_molecule"])

    ########## for optimization ##########
    parser.add_argument("--T_list", type=int, nargs="+", default=[4, 10, 20])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--mol_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=0.1)
    parser.add_argument("--mol_lr_scale", type=float, default=0.1)
    parser.add_argument("--decay", type=float, default=0)

    ########## for contrastive objective ##########
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    ########## for BERT model ##########
    parser.add_argument("--max_seq_len", type=int, default=512)

    ########## for molecule model ##########
    parser.add_argument("--molecule_type", type=str, default="SMILES", choices=["SMILES", "Graph", "Motif"])

    ########## for MegaMolBART ##########
    parser.add_argument("--vocab_path", type=str, default="../MoleculeSTM/bart_vocab.txt")

    ########## for 2D GNN ##########
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    ########## for saver ##########
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--input_model_dir", type=str, default=None)
    parser.add_argument("--input_model_path", type=str, default=None)
    parser.add_argument("--output_model_dir", type=str, default=None)

    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    parser.add_argument("--last_epoch", type=str, default=None)
    args = parser.parse_args()
    print("arguments\t", args)
    torch.multiprocessing.set_sharing_strategy('file_system')

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    ##### prepare text model #####
    ##### by default, this is load_mode_1 #####
    if args.text_type == "SciBERT":
        pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'pretrained_SciBERT')
        text_tokenizer = AutoTokenizer.from_pretrained('../data/pretrained_SciBERT', )
        text_dim = 768

    config = BertConfig.from_json_file(args.config_file)

    ##### prepare molecule model #####
    if args.molecule_type == "SMILES":
        if args.model_loading_mode == "load_from_latest":
            input_model_path = os.path.join(args.input_model_dir, "molecule_model.pth")
            print("Loading from {}...".format(input_model_path))
            MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=None, output_dir=None)
            molecule_model = MegaMolBART_wrapper.model
            state_dict = torch.load(input_model_path, map_location='cpu')
            molecule_model.load_state_dict(state_dict)
        elif args.model_loading_mode == "load_mode_0":
            MegaMolBART_wrapper = MegaMolBART(vocab_path=args.vocab_path, input_dir=None, output_dir=None)
            molecule_model = MegaMolBART_wrapper.model
            print("Random init for MegaMolBART.")
        elif args.model_loading_mode == "load_mode_1":
            # This is loading from the pretarined_MegaMolBART
            # --input_model_dir=../data/pretrained_MegaMolBART/checkpoints
            MegaMolBART_wrapper = MegaMolBART(input_dir="../data/pretrained_MegaMolBART/checkpoints", output_dir=None)
            molecule_model = MegaMolBART_wrapper.model
            print("Loading from ../data/pretrained_MegaMolBART/checkpoint.")
        molecule_dim = 768
    elif args.molecule_type == "Motif":
        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)
        molecule_model = GNN_motifpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model)
        molecule_dim = args.gnn_emb_dim
        input_model_path = os.path.join("checkpoints/{}".format(args.input_model_dir),
                                        "molecule_model_{}.pth".format(args.last_epoch))
        print("Loading from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location="cpu")
        molecule_model.load_state_dict(state_dict)
        molecule_model = molecule_model.to(device)
    else:
        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)
        molecule_model = GNN_graphpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model)
        molecule_dim = config.v_hidden_size
        if args.model_loading_mode == "load_from_latest":
            input_model_path = os.path.join(args.input_model_dir, "molecule_model.pth")
            print("Loading from {}...".format(input_model_path))
            state_dict = torch.load(input_model_path, map_location='cpu')
            molecule_model.load_state_dict(state_dict)
        elif args.model_loading_mode == "load_mode_0":
            print("Random init for GNN.")
        elif args.model_loading_mode == "load_mode_1":
            print("Loading from ../data/pretrained_GraphMVP/GraphMVP_G/model.pth")
            molecule_model.from_pretrained("../data/pretrained_GraphMVP/GraphMVP_G/model.pth")

    # Rewrite the seed by MegaMolBART
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    mol2latent = nn.Linear(molecule_dim, config.v_hidden_size).to(device)
    input_model_path = os.path.join("checkpoints/{}".format(args.input_model_dir), "mol2latent_model_{}.pth".format(args.last_epoch))
    print("Loading from {}...".format(input_model_path))
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)


    model = BertForMultiModalPreTraining(config).to(device)
    input_model_path = os.path.join("checkpoints/{}".format(args.input_model_dir), "multi_model_{}.pth".format(args.last_epoch))
    print("Loading from {}...".format(input_model_path))
    loaded_state_dict = torch.load(input_model_path, map_location='cpu')
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if
                               k in model_state_dict and model_state_dict[k].size() == loaded_state_dict[k].size()}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

    molecule_model = molecule_model.to(device)
    mol2latent = mol2latent.to(device)

    T_max = max(args.T_list) - 1

    initial_test_acc_list = []
    test_mode = args.test_mode
    dataset_folder = os.path.join(args.dataspace_path, "DrugBank_data")
    if args.molecule_type == "SMILES":
        dataset_class = DrugBank_Datasets_SMILES_retrieval
        dataloader_class = torch_DataLoader

        if args.task == "molecule_description":
            template = "SMILES_description_{}.txt"
        elif args.task == "molecule_description_removed_PubChem":
            template = "SMILES_description_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_description_Raw":
            template = "SMILES_description_{}_Raw.txt"
        elif args.task == "molecule_description_removed_PubChem_Raw":
            template = "SMILES_description_removed_from_PubChem_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics":
            template = "SMILES_pharmacodynamics_{}.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_pharmacodynamics_Raw":
            template = "SMILES_pharmacodynamics_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem_Raw":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}_Raw.txt"

        full_dataset = dataset_class(dataset_folder, 'full', neg_sample_size=T_max, template=template)

    else:
        dataset_class = DrugBank_Datasets_Graph_retrieval
        dataloader_class = pyg_DataLoader
        processed_dir_prefix = args.task

        if args.task == "molecule_description":
            template = "SMILES_description_{}.txt"
        elif args.task == "molecule_description_removed_PubChem":
            template = "SMILES_description_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_description_Raw":
            template = "SMILES_description_{}_Raw.txt"
        elif args.task == "molecule_description_removed_PubChem_Raw":
            template = "SMILES_description_removed_from_PubChem_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics":
            template = "SMILES_pharmacodynamics_{}.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}.txt"
        elif args.task == "molecule_pharmacodynamics_Raw":
            template = "SMILES_pharmacodynamics_{}_Raw.txt"
        elif args.task == "molecule_pharmacodynamics_removed_PubChem_Raw":
            template = "SMILES_pharmacodynamics_removed_from_PubChem_{}_Raw.txt"

        full_dataset = dataset_class(dataset_folder, 'full', neg_sample_size=T_max, processed_dir_prefix=processed_dir_prefix, template=template)

    full_dataloader = dataloader_class(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # The program will get blcoked with none-zero num_workers

    initial_test_acc_list_t, initial_test_acc_list_m  = eval_epoch(full_dataloader)
    # print('Initial', initial_test_acc_list_t)

    row_t = ", ".join(["{:.4f}".format(x * 100) for x in initial_test_acc_list_t])
    row_m = ", ".join(["{:.4f}".format(x * 100) for x in initial_test_acc_list_m])
    print("results t,", row_t, "results m", row_m)
    str_content = "Des: results t, {}, results m, {}\n".format(row_t, row_m)
    sys.exit()