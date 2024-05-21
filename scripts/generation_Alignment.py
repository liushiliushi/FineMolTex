import argparse
import os
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

from FineMolTex.utils import get_molecule_repr_MoleculeSTM
from FineMolTex.models import MLP
from FineMolTex.downstream_molecule_edit_utils import load_molecule_models, load_molecule_models2
from MoleculeSTM.utils import freeze_network
from FineMolTex.datasets import ZINC250K_Dataset_SMILES, ZINC250K_Dataset_Graph,  ZINC250K_Dataset_GraphMotif, ZINC250K_Dataset_GraphMotifSub
from FineMolTex.models import GNN, GNN_graphpred, GNN_motifpred, BertConnectionLayer, BertPreTrainingHeads, BertForMultiModalPreTraining,  BertConfig
from FineMolTex.utils import prepare_text_tokens, get_molecule_repr_MoleculeSTM, get_molecule_repr_MoleculeSTM2, freeze_network


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    if args.SSL_loss == 'EBM_NCE':

        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)] for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))
        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T
        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))

        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        SSL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        SSL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        SSL_acc = SSL_acc.detach().cpu().item()
        
    elif args.SSL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        SSL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        SSL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    
    elif args.SSL_loss == 'RR':
        criterion = nn.MSELoss()
        SSL_loss = criterion(X, Y)
        SSL_acc = 0

    else:
        raise Exception

    return SSL_loss, SSL_acc

def mean_pooling(token_embeddings, attention_mask):
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [pad, B, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 0) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(0), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask


def get_molecule_repr_generation(molecule_data, molecule_model, molecule_type="MegaMolBART", MegaMolBART_wrapper=None):
    if molecule_type == "MegaMolBART":
        embedding, pad_mask = MegaMolBART_wrapper.smileslist2embedding_model_given(molecule_model, molecule_data)  # [pad, B, d], [pad, B]
        molecule_repr = mean_pooling(embedding, pad_mask)
    else:
        molecule_repr, _ = molecule_model(molecule_data)
    return molecule_repr


def save_model(save_best, epoch=None):

    if args.output_model_dir is not None:
        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            model_file = "model.pth"

        elif epoch is None:
            model_file = "model_final.pth"

        else:
            model_file = "model_{}.pth".format(epoch)

        save_path = "checkpoints/{}".format(args.output_model_dir)
        if not os.path.exists(save_path):
            # 如果目录不存在，则创建它
            os.makedirs(save_path)
        saved_file_path = os.path.join(save_path, "generation2MoleculeSTM_{}".format(model_file))
        torch.save(generation2MoleculeSTM.state_dict(), saved_file_path)
        
        saved_file_path = os.path.join(save_path, "MoleculeSTM2generation_{}".format(model_file))
        torch.save(MoleculeSTM2generation.state_dict(), saved_file_path)
    return


def train(epoch):
    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader
    
    start_time = time.time()
    accum_loss, accum_acc = 0, 0
    step = 0
    for batch in L:
        if args.MoleculeSTM_molecule_type == "SMILES":
            SMILES_list = batch
        else:
            SMILES_list, graph = batch
            graph = graph.to(args.device)

        if args.MoleculeSTM_molecule_type == "SMILES":
            molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM(
                SMILES_list, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper
            )
            molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)

        elif args.MoleculeSTM_molecule_type == "Motif":
            molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM2(
                graph, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM, args=args, model=model_MoleculeSTM,
                molecule_type="Motif", MegaMolBART_wrapper=None)

            molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)

        else:
            molecule_repr_MoleculeSTM = get_molecule_repr_MoleculeSTM(
                graph, molecule_model=molecule_model_MoleculeSTM, mol2latent=mol2latent_MoleculeSTM,
                molecule_type=args.MoleculeSTM_molecule_type, MegaMolBART_wrapper=None
            )
            molecule_repr_MoleculeSTM2generation = MoleculeSTM2generation(molecule_repr_MoleculeSTM)

        if args.generation_model == "MegaMolBART":
            molecule_repr_generation = get_molecule_repr_generation(
                SMILES_list, molecule_model=molecule_model_generation,
                molecule_type="MegaMolBART", MegaMolBART_wrapper=MegaMolBART_wrapper
            )
        else:  # for HierVAE
            hiervae_data_list = MolGraph.tensorize(SMILES_list, vocab, avocab)
            molecule_repr_generation = molecule_model_generation.forward_MoleculeSTM(hiervae_data_list)
        molecule_repr_generation2MoleculeSTM = generation2MoleculeSTM(molecule_repr_generation)

        loss_01, acc_01 = do_CL(molecule_repr_generation, molecule_repr_MoleculeSTM2generation, args)
        loss_02, acc_02 = do_CL(molecule_repr_MoleculeSTM, molecule_repr_generation2MoleculeSTM, args)
        loss = (loss_01 + loss_02) / 2
        acc = (acc_01 + acc_02) / 2
        print("loss:{} acc:{}".format(loss, acc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accum_loss += loss.item()
        accum_acc += acc
        step += 1
        if step % 50 == 0 and step!= 0:
            save_model(save_best=False, epoch=epoch)

    accum_loss /= len(L)
    accum_acc /= len(L)
    
    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True, epoch=epoch)
    print("SSL Loss: {:.5f}\tSSL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="ZINC250K")
    parser.add_argument("--MoleculeSTM_molecule_type", type=str, default="SMILES", choices=["SMILES", "Graph","Motif"])
    parser.add_argument("--output_model_dir", type=str, default=None)

    ########## for MoleculeSTM ##########
    parser.add_argument("--MoleculeSTM_model_dir", type=str, default="../../pretrained_model")
    parser.add_argument("--SSL_emb_dim", type=int, default=768)
    ########## for 2D GNN ##########
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    ########## for generation ##########
    parser.add_argument('--generation_model', type=str, default="MegaMolBART", choices=["MegaMolBART"])

    ######### for MegaMolBART ##########
    parser.add_argument("--MegaMolBART_generation_model_dir", type=str, default="../data/pretrained_MegaMolBART/checkpoints")
    parser.add_argument("--vocab_path", type=str, default="MoleculeSTM/bart_vocab.txt")

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--generation_lr", type=float, default=1e-2)
    parser.add_argument("--MoleculeSTM_lr", type=float, default=1e-2)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE", "RR"])
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument('--use_normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.add_argument("--input_model_dir", type=str, default=None)
    parser.add_argument("--input_model_path", type=str, default=None)

    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    parser.add_argument("--last_epoch", type=str, default=None)

    parser.set_defaults(normalize=True)

    args = parser.parse_args()
    print(args)
    config = BertConfig.from_json_file(args.config_file)
    args.max_seq_len = config.max_position_embeddings

    
    if args.generation_model == "MegaMolBART":
        if args.MoleculeSTM_molecule_type == "SMILES":
            if args.dataset == "ZINC250K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_SMILES(dataset_root)
            elif args.dataset == "ZINC250K1K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_SMILES(dataset_root, 1000)
            elif args.dataset == "ZINC250K10K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_SMILES(dataset_root, 10000)
            else:
                raise Exception
            dataloader_class = torch_DataLoader
        elif args.MoleculeSTM_molecule_type == "Motif":
            if args.dataset == "ZINC250K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_GraphMotif(dataset_root)
            elif args.dataset == "ZINC250K1K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_GraphMotif(dataset_root, 1000)
            elif args.dataset == "ZINC250K10K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_Graph(dataset_root, 10000)
            else:
                raise Exception
            dataloader_class = pyg_DataLoader
        else:
            if args.dataset == "ZINC250K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_Graph(dataset_root)
            elif args.dataset == "ZINC250K1K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_Graph(dataset_root, 100)
            elif args.dataset == "ZINC250K10K":
                dataset_root = os.path.join(args.dataspace_path, "ZINC250K_data")
                dataset = ZINC250K_Dataset_Graph(dataset_root, 10000)
            else:
                raise Exception
            dataloader_class = pyg_DataLoader
    else:
        raise NotImplementedError

    # args.device = torch.device("cpu")
    args.device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    MegaMolBART_wrapper, molecule_model_generation, molecule_dim_generation, \
        molecule_model_MoleculeSTM, mol2latent_MoleculeSTM, molecule_dim_MoleculeSTM, model_MoleculeSTM = load_molecule_models2(args, config)

    # MegaMolBART_wrapper.device = device
    # molecule_dim_generation = molecule_dim_generation.to(device)
    # molecule_model_generation = molecule_model_generation.to(device)
    # molecule_model_MoleculeSTM = molecule_model_MoleculeSTM.to(args.device)
    # mol2latent_MoleculeSTM = mol2latent_MoleculeSTM.to(args.device)

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    freeze_network(molecule_model_generation)
    freeze_network(mol2latent_MoleculeSTM)
    freeze_network(molecule_model_MoleculeSTM)
    molecule_model_generation.eval()
    mol2latent_MoleculeSTM.eval()
    molecule_model_MoleculeSTM.eval()
    model_MoleculeSTM.eval()
    
    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    generation2MoleculeSTM = MLP(molecule_dim_generation, [molecule_dim_MoleculeSTM, molecule_dim_MoleculeSTM]).to(args.device)
    MoleculeSTM2generation = MLP(molecule_dim_MoleculeSTM, [molecule_dim_generation, molecule_dim_generation]).to(args.device)

    # input_model_path = os.path.join("checkpoints/{}".format(args.output_model_dir),
    #                                 "generation2MoleculeSTM_model_{}.pth".format(args.last_epoch2))
    # print("Loading from {}...".format(input_model_path))
    # state_dict = torch.load(input_model_path, map_location="cpu")
    # generation2MoleculeSTM.load_state_dict(state_dict)
    # generation2MoleculeSTM = generation2MoleculeSTM.to(args.device)
    #
    # input_model_path = os.path.join("checkpoints/{}".format(args.output_model_dir),
    #                                 "MoleculeSTM2generation_model_{}.pth".format(args.last_epoch2))
    # print("Loading from {}...".format(input_model_path))
    # state_dict = torch.load(input_model_path, map_location="cpu")
    # MoleculeSTM2generation.load_state_dict(state_dict)
    # MoleculeSTM2generation = MoleculeSTM2generation.to(args.device)

    model_param_group = [
        {"params": generation2MoleculeSTM.parameters(), "lr": args.generation_lr},
        {"params": MoleculeSTM2generation.parameters(), "lr": args.MoleculeSTM_lr},
    ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10
    
    for e in range(1, args.epochs+1):
        print("Epoch {}".format(e))
        train(e)
