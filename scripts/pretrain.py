import os
import time
import numpy as np
from tqdm import tqdm
import argparse
from collections import Counter
import sys
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader as torch_DataLoader
import json
from torch_geometric.loader import DataLoader as pyg_DataLoader
from transformers import AutoModel, AutoTokenizer, VisualBertModel
import matplotlib.pyplot as plt
from FineMolTex.datasets import (
    PubChemSTM_Datasets_SMILES, PubChemSTM_SubDatasets_SMILES,
    PubChemSTM_Datasets_Graph, PubChemSTM_Datasets_GraphMotif, PubChemSTM_SubDatasets_Graph,
    PubChemSTM_SubDatasets_GraphMotif,
    PubChemSTM_Datasets_Raw_SMILES, PubChemSTM_SubDatasets_Raw_SMILES,
    PubChemSTM_Datasets_Raw_Graph, PubChemSTM_SubDatasets_Raw_Graph,
    DrugBank_Datasets_Graph_ATC_Sub, DrugBank_Datasets_Graph_retrieval_Sub, PubChemSTM_Datasets_Motif2
)
from FineMolTex.models import GNN, GNN_graphpred, GNN_motifpred, GNN_motifpred2, GNN_motifpred3, BertConnectionLayer, \
    BertPreTrainingHeads, BertForMultiModalPreTraining, BertConfig
from FineMolTex.utils import prepare_text_tokens, get_molecule_repr_MoleculeSTM, freeze_network, prepare_text_tokens, \
    get_molecule_repr_MoleculeSTM, get_molecule_repr_MoleculeSTM2, do_CL_eval, get_token_importance
from FineMolTex.models.mega_molbart.mega_mol_bart import MegaMolBART
import json
import time
import os
import torch.autograd
from accelerate import Accelerator
from accelerate.utils import set_seed

torch.autograd.set_detect_anomaly(True)
result_dir = "F0206"


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
        CL_loss = (loss_pos + args.CL_neg_samples * loss_neg) / (1 + args.CL_neg_samples)

        CL_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.SSL_loss == 'InfoNCE':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    else:
        raise Exception

    return CL_loss, CL_acc


def save_model(save_best, epoch=None):
    if args.output_model_dir is not None and accelerator.is_main_process:
        save_path = "checkpoints/{}".format(args.output_model_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 获取未包装的模型
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_molecule_model = accelerator.unwrap_model(molecule_model)
        unwrapped_mol2latent = accelerator.unwrap_model(mol2latent)
        unwrapped_frag_classifier = accelerator.unwrap_model(frag_classifier)

        if save_best:
            global optimal_loss
            print("save model with loss: {:.5f}".format(optimal_loss))
            model_file = "model.pth"
            model_file_past = "model.pth"

        elif epoch is None:
            model_file = "model_final.pth"
            model_file_past = "model_.pth".format()

        else:
            model_file = "model_{}.pth".format(epoch)
            model_file_past = "model_{}.pth".format(epoch - 1)

        saved_file_path = os.path.join(save_path, "molecule_{}".format(model_file))
        if os.path.exists(os.path.join(save_path, "molecule_{}".format(model_file_past))):
            os.remove(os.path.join(save_path, "molecule_{}".format(model_file_past)))
        torch.save(unwrapped_molecule_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(save_path, "mol2latent_{}".format(model_file))
        if os.path.exists(os.path.join(save_path, "mol2latent_{}".format(model_file_past))):
            os.remove(os.path.join(save_path, "mol2latent_{}".format(model_file_past)))
        torch.save(unwrapped_mol2latent.state_dict(), saved_file_path)

        saved_file_path = os.path.join(save_path, "multi_{}".format(model_file))
        if os.path.exists(os.path.join(save_path, "multi_{}".format(model_file_past))):
            os.remove(os.path.join(save_path, "multi_{}".format(model_file_past)))
        torch.save(unwrapped_model.state_dict(), saved_file_path)

        saved_file_path = os.path.join(save_path, "frag_{}".format(model_file))
        if os.path.exists(os.path.join(save_path, "frag_{}".format(model_file_past))):
            os.remove(os.path.join(save_path, "frag_{}".format(model_file_past)))
        torch.save(unwrapped_frag_classifier.state_dict(), saved_file_path)
    return


def create_molecule_masks(molecule_repr, molecule_data, attention_weights, args, device):
    """
    为分子数据创建mask

    Args:
        molecule_repr: 分子表示
        molecule_data: 分子数据
        attention_weights: 注意力权重
        args: 参数配置
        device: 计算设备

    Returns:
        tuple: (masked_tensor, mask_positions, mask_labels)
    """
    masklabels = []
    masklist = []
    padded_tensor_m = torch.zeros((len(molecule_repr), args.max_seq_len, args.SSL_emb_dim)).to(device)

    for num, item in enumerate(molecule_repr):
        tokens = molecule_data.motiflabel[num]
        if not tokens:
            continue

        is_long_seq = item.size(0) > (args.max_seq_len - 1)
        mask_ratio = 0.15 if is_long_seq else 0.2

        weights = attention_weights[num][1:len(tokens) + 1].detach().cpu().numpy()

        available_maskids = molecule_data['maskids'][num]

        if not available_maskids:
            continue

        token_weights = weights[available_maskids]
        token_weights = np.exp(token_weights) / np.sum(np.exp(token_weights))
        num_masks = max(1, int(len(available_maskids) * mask_ratio))

        # 选择重要的tokens进行mask
        important_maskids = np.random.choice(
            available_maskids,
            size=min(num_masks, len(available_maskids)),
            replace=False,
            p=token_weights
        ).tolist()

        # 如果需要补充随机tokens
        if len(important_maskids) < num_masks:
            remaining = list(set(available_maskids) - set(important_maskids))
            if remaining:
                additional = np.random.choice(
                    remaining,
                    size=min(num_masks - len(important_maskids), len(remaining)),
                    replace=False
                )
                important_maskids.extend(additional)

        if important_maskids:
            padded_tensor_m[num] = padded_tensor_m[num].clone()
            padded_tensor_m[num, 1:item.size(0) + 1] = item
            padded_tensor_m[num, torch.tensor(important_maskids).long() + 1] = 0
            masklist.extend([(i + 1 + args.max_seq_len * num) for i in important_maskids])
            masklabels.extend([molecule_data['motiflabel'][num][i] for i in important_maskids])

    if not masklist:
        return None, None, None

    return padded_tensor_m, masklist, masklabels


def create_text_masks(description_tokens_ids, description_masks, attention_weights_text, text_tokenizer):
    """
    为文本数据创建mask

    Args:
        description_tokens_ids: 文本token IDs
        attention_weights_text: 文本注意力权重
        text_tokenizer: 文本分词器

    Returns:
        tuple: (text_masks, masked_tokens, mask_labels)
    """
    text_masks = torch.zeros_like(description_tokens_ids, dtype=torch.bool)
    special_tokens = [
        text_tokenizer.cls_token_id,
        text_tokenizer.sep_token_id,
        text_tokenizer.pad_token_id
    ]

    for i in range(len(description_tokens_ids)):
        tokens = description_tokens_ids[i]
        valid_token_positions = torch.where(description_masks[i][1:] == 1)[0] + 1  # +1 是为了补偿排除的CLS位置

        if valid_token_positions.numel() > 0:
            weights = attention_weights_text[i][valid_token_positions].detach().cpu().numpy()
            weights = np.exp(weights) / np.sum(np.exp(weights))
            num_masks = max(1, int(len(valid_token_positions) * 0.2))

            mask_positions = np.random.choice(
                valid_token_positions.cpu().numpy(),
                size=min(num_masks, len(valid_token_positions)),
                replace=False,
                p=weights
            )
            text_masks[i][mask_positions] = True

    masked_tokens = description_tokens_ids.clone()
    mask_labels = description_tokens_ids.clone()
    mask_labels[~text_masks] = -1
    masked_tokens[text_masks] = text_tokenizer.mask_token_id

    return text_masks, masked_tokens, mask_labels


def train(
        epoch,
        dataloader,
        wandb_run,
        text_tokenizer,
        model,
        molecule_model, MegaMolBART_wrapper=None):
    if args.representation_frozen:
        molecule_model.eval()
    else:
        molecule_model.train()
    model.train()
    mol2latent.train()
    frag_classifier.train()

    if args.verbose:
        L = tqdm(dataloader)
    else:
        L = dataloader

    start_time = time.time()
    accum_loss, accum_acc = 0, 0
    Closs = torch.nn.CrossEntropyLoss(ignore_index=-1)
    criterion_frag = nn.CrossEntropyLoss()
    motif_step_loss = 0
    lm_step_loss = 0
    next_step_loss = 0
    frag_step_loss = 0
    frag_step_loss2 = 0

    dict = []

    # 打开文件并读取每一行
    with open('../data/PubChemSTM_data/vocab2.txt', 'r') as file:
        for line in file:
            # 将每一行添加到列表中
            dict.append(line.strip())

    for step, batch in enumerate(L):
        try:
            description = batch[0]
            molecule_data = batch[1]

            molecule_repr = get_molecule_repr_MoleculeSTM(
                molecule_data, mol2latent=mol2latent,
                molecule_type=molecule_type, molecule_model=molecule_model,
                device=device
            )

            description_tokens_ids, description_masks, = prepare_text_tokens(
                device=device, description=description, tokenizer=text_tokenizer, max_seq_len=args.max_seq_len,
                mask=None)
            description_tokens_ids_m, description_masks_m, token_masks, mask_labels = prepare_text_tokens(
                device=device, description=description, tokenizer=text_tokenizer, max_seq_len=args.max_seq_len,
                mask=0.15)

            d_attention_mask = description_masks
            d_attention_mask_m = description_masks_m

            if molecule_type == "SMILES":
                molecule_data = list(molecule_data)
                molecule_repr = get_molecule_repr_MoleculeSTM(
                    molecule_data, mol2latent=mol2latent,
                    molecule_type=molecule_type, MegaMolBART_wrapper=MegaMolBART_wrapper, device=args.device)
            elif molecule_type == "Motif":
                molecule_data = molecule_data.to(device)
                molecule_repr = get_molecule_repr_MoleculeSTM(
                    molecule_data, mol2latent=mol2latent,
                    molecule_type=molecule_type, molecule_model=molecule_model, device=args.device)
                m_attention_mask = torch.full((len(molecule_repr), args.max_seq_len), float(0)).to(args.device)
                padded_tensor = torch.zeros((len(molecule_repr), args.max_seq_len, args.SSL_emb_dim)).to(args.device)
                m_position_embedding = torch.zeros((len(molecule_repr), args.max_seq_len, 1)).to(args.device)

                num = 0
                for item in molecule_repr:
                    if item.size(0) > (args.max_seq_len - 1):
                        padded_tensor[num, 1:args.max_seq_len, :] = item[:(args.max_seq_len - 1)]
                        m_position_embedding[num, :args.max_seq_len] = 1
                        m_attention_mask[num, :args.max_seq_len] = 1
                    else:
                        padded_tensor[num, 1:item.size(0) + 1, :] = item
                        m_attention_mask[num, :item.size(0) + 1] = 1
                        m_position_embedding[num, 1:item.size(0) + 1, :] = torch.tensor(
                            molecule_data.positions[num]).unsqueeze(-1)
                        m_attention_mask[num, :item.size(0) + 1] = 1
                    num = num + 1

                m_repr = padded_tensor
            else:
                molecule_data = molecule_data.to(device)
                molecule_repr = get_molecule_repr_MoleculeSTM(
                    molecule_data, mol2latent=mol2latent,
                    molecule_type=molecule_type, molecule_model=molecule_model, device=args.device)

            # ==================contrastive==================
            model.bert.encoder.with_coattention = False
            prediction_scores_t2, prediction_scores_m2, all_attention_mask2, pooled_output_t2, pooled_output_m2, sequence_output_t2, sequence_output_m2 = model(
                description_tokens_ids,
                m_repr,
                d_attention_mask,
                m_attention_mask,
                m_position_embedding=m_position_embedding.clone(),
                masked_lm_labels=None,
                image_label=None,
                image_target=None,
                next_sentence_label=None,
            )

            next_loss1, next_acc1 = do_CL(pooled_output_t2, pooled_output_m2, args)
            next_loss2, next_acc2 = do_CL(pooled_output_m2, pooled_output_t2, args)
            next_loss = (next_loss1 + next_loss2) / 2
            next_acc = (next_acc1 + next_acc2) / 2

            # ============mask================
            if True:
                model.bert.encoder.with_coattention = True
                attention_weights = get_token_importance(sequence_output_m2, pooled_output_m2)
                attention_weights_text = get_token_importance(sequence_output_t2, pooled_output_t2)

                # 创建分子masks
                m_repr_m, masklist, masklabels = create_molecule_masks(
                    molecule_repr,
                    molecule_data,
                    attention_weights,
                    args,
                    device
                )

                if m_repr_m is None:
                    continue
                frag_pos = torch.tensor(masklist).to(device)
                batch_frag = torch.tensor(masklabels).long().to(device)

                # 创建文本masks
                text_masks, description_tokens_ids_m, mask_labels = create_text_masks(
                    description_tokens_ids,
                    description_masks,
                    attention_weights_text,
                    text_tokenizer
                )

                model.bert.encoder.with_coattention = True
                prediction_scores_t, prediction_scores_m, all_attention_mask, pooled_output_t, pooled_output_m, sequence_output_t, sequence_output_m = model(
                    description_tokens_ids_m,
                    m_repr,
                    d_attention_mask_m,
                    m_attention_mask,
                    m_position_embedding=m_position_embedding.clone(),
                    masked_lm_labels=None,
                    image_label=None,
                    image_target=None,
                    next_sentence_label=None,
                )

                frag_pred = frag_classifier(
                    torch.index_select(sequence_output_m.reshape(-1, sequence_output_m.shape[2]), 0, frag_pos))
                frag_loss = criterion_frag(frag_pred, batch_frag)
                frag_pred_label = torch.argmax(frag_pred, dim=1)

                if frag_pred_label.size()[0] != batch_frag.size()[0]:
                    stop = 1
                frag_acc = torch.eq(frag_pred_label, batch_frag).sum().item() / batch_frag.size()[0]
                masked_lm_loss = Closs(
                    prediction_scores_t[text_masks].view(-1, prediction_scores_t.shape[2]),
                    mask_labels[text_masks].to(args.device),  # 只取被mask的位置的标签
                )
            else:
                masked_lm_loss = torch.tensor(0)
                frag_loss = torch.tensor(0)
                motif_acc = 0
                frag_acc = 0
            loss = 1 * next_loss + args.alpha * masked_lm_loss + args.beta * frag_loss
            frag_step_loss = frag_step_loss + frag_loss.item()
            if not masked_lm_loss.isnan().any():
                lm_step_loss = lm_step_loss + masked_lm_loss.item()
            next_step_loss = next_step_loss + next_loss.item()

            accelerator.print(
                "{} {} motif_loss: {} motif_acc:{} lm_loss:{} next_loss: {} next_acc:{}".format(epoch, step,
                                                                                                frag_loss.item(),
                                                                                                frag_acc,
                                                                                                masked_lm_loss.item(),
                                                                                                next_loss.item(),
                                                                                                next_acc))

            if args.wandb and accelerator.is_main_process:
                wandb_run.log({
                    "total_loss": loss.item(),
                    "next_loss": next_loss.item(),
                    "masked_lm_loss": masked_lm_loss.item(),
                    "frag_loss": frag_loss.item(),
                    "frag_accuracy": frag_acc,
                    "next_accuracy": next_acc
                })

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            accum_loss += loss.item()


        except RuntimeError as e:
            print(f"Error in batch {step}: {str(e)}")
            print(f"Batch size: {len(batch[0])}")
            print(f"Molecule data shape: {molecule_data.x.shape if hasattr(molecule_data, 'x') else 'No x attribute'}")
            continue  # 跳过这个批次

    save_model(False, epoch)
    accum_loss /= len(L)
    accum_acc /= len(L)

    global optimal_loss
    temp_loss = accum_loss
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print("CL Loss: {:.5f}\tCL Acc: {:.5f}\tTime: {:.5f}".format(accum_loss, accum_acc, time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dataspace_path", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="PubChemSTM")
    parser.add_argument("--text_type", type=str, default="SciBERT", choices=["SciBERT"])
    parser.add_argument("--molecule_type", type=str, default="SMILES", choices=["SMILES", "Graph", "Motif"])
    parser.add_argument("--representation_frozen", dest='representation_frozen', action='store_true')
    parser.add_argument('--no_representation_frozen', dest='representation_frozen', action='store_false')
    parser.set_defaults(representation_frozen=False)
    parser.add_argument(
        "--config_file",
        default="config/config.json",
        type=str,
        # required=True,
        help="The config file which specified the model details.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--text_lr", type=float, default=1e-4)
    parser.add_argument("--mol_lr", type=float, default=1e-5)
    parser.add_argument("--co_lr", type=float, default=1e-4)
    parser.add_argument("--pre_lr", type=float, default=1e-4)
    # parser.add_argument("--text_lr_scale", type=float, default=1)
    parser.add_argument("--mol_lr_scale", type=float, default=3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)
    parser.add_argument("--input_model_dir", type=str, default=None)
    parser.add_argument("--last_epoch", type=str, default=None)
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--wandb", type=bool, default=False)

    ########## for SciBERT ##########
    parser.add_argument("--max_seq_len", type=int, default=512)

    ########## for MegaMolBART ##########
    parser.add_argument("--megamolbart_input_dir", type=str, default="../data/pretrained_MegaMolBART/checkpoints")
    parser.add_argument("--vocab_path", type=str, default="../MoleculeSTM/bart_vocab.txt")
    parser.add_argument("--datafile", type=str, default="geometric_data_motif_processed.pt")

    ########## for 2D GNN ##########
    parser.add_argument("--pretrain_gnn_mode", type=str, default="GraphMVP_G", choices=["GraphMVP_G"])
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    ########## for contrastive SSL ##########
    parser.add_argument("--SSL_loss", type=str, default="EBM_NCE", choices=["EBM_NCE", "InfoNCE"])
    parser.add_argument("--SSL_emb_dim", type=int, default=768)
    parser.add_argument("--CL_neg_samples", type=int, default=4)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    args = parser.parse_args()
    print("arguments\t", args)

    accelerator = Accelerator(
        mixed_precision='no',  # 使用混合精度训练
        gradient_accumulation_steps=1,
        log_with="wandb" if args.wandb else None
    )
    device = accelerator.device
    args.device = device
    config = BertConfig.from_json_file(args.config_file)
    if args.wandb and accelerator.is_main_process:
        wandb_run = wandb.init(
            project="MoleculeSTM",  # 项目名称
            name=args.output_model_dir,  # 实验名称
            config={  # 配置参数
                "batch_size": args.batch_size,
                "learning_rate": args.text_lr,
                "mol_lr": args.mol_lr,
                "epochs": args.epochs,
                "model_type": args.molecule_type,
                "SSL_loss": args.SSL_loss,
            }
        )
    else:
        wandb_run = None
    args.max_seq_len = config.max_position_embeddings
    args.alpha = config.alpha
    args.beta = config.beta
    with open(args.config_file, 'r') as source_file:
        data = json.load(source_file)

    # Converting the JSON data to a string with indentation for readability
    # data_string = json.dumps(data, indent=4)
    #
    # # Writing data to a text file
    # with open('images/testsbash/{}.txt'.format(args.output_model_dir), 'w') as destination_file:
    #     destination_file.write(data_string)
    #     destination_file.write('\n')
    #     destination_file.write('without mask\n')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if "PubChemSTM" in args.dataset:
        dataset_root = os.path.join(args.dataspace_path, "PubChemSTM_data")
    else:
        raise Exception

    kwargs = {}

    results = []

    ##### prepare molecule model #####
    molecule_type = args.molecule_type
    if molecule_type == "SMILES":
        if args.dataset == "PubChemSTM":
            dataset = PubChemSTM_Datasets_SMILES(dataset_root)
        elif args.dataset == "PubChemSTM1K":
            dataset = PubChemSTM_SubDatasets_SMILES(dataset_root, size=1000)
        elif args.dataset == "PubChemSTM10K":
            dataset = PubChemSTM_SubDatasets_SMILES(dataset_root, size=10000)
        elif args.dataset == "PubChemSTM_Raw":
            dataset = PubChemSTM_Datasets_Raw_SMILES(dataset_root)
        elif args.dataset == "PubChemSTM1K_Raw":
            dataset = PubChemSTM_SubDatasets_Raw_SMILES(dataset_root, size=1000)
        elif args.dataset == "PubChemSTM10K_Raw":
            dataset = PubChemSTM_SubDatasets_Raw_SMILES(dataset_root, size=10000)
        else:
            raise Exception
        dataloader_class = torch_DataLoader

        if args.output_model_dir is not None:
            MegaMolBART_dir = os.path.join(args.output_model_dir, "MegaMolBART")
        else:
            MegaMolBART_dir = None
        MegaMolBART_wrapper = MegaMolBART(
            vocab_path=args.vocab_path,
            input_dir=args.megamolbart_input_dir,
            output_dir=MegaMolBART_dir)
        molecule_model = MegaMolBART_wrapper.model
        kwargs["MegaMolBART_wrapper"] = MegaMolBART_wrapper
        kwargs["molecule_model"] = molecule_model
        molecule_dim = args.SSL_emb_dim

    elif molecule_type == "Graph":
        if args.dataset == "PubChemSTM":
            dataset = PubChemSTM_Datasets_Graph(dataset_root)
        elif args.dataset == "PubChemSTM1K":
            dataset = PubChemSTM_SubDatasets_Graph(dataset_root, size=100)
        elif args.dataset == "PubChemSTM10K":
            dataset = PubChemSTM_SubDatasets_Graph(dataset_root, size=10000)
        elif args.dataset == "PubChemSTM_Raw":
            dataset = PubChemSTM_Datasets_Raw_Graph(dataset_root)
        elif args.dataset == "PubChemSTM1K_Raw":
            dataset = PubChemSTM_SubDatasets_Raw_Graph(dataset_root, size=1000)
        elif args.dataset == "PubChemSTM10K_Raw":
            dataset = PubChemSTM_SubDatasets_Raw_Graph(dataset_root, size=10000)
        dataloader_class = pyg_DataLoader
        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)
        molecule_model = GNN_graphpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model)
        pretrained_model_path = os.path.join(args.dataspace_path, "pretrained_GraphMVP", args.pretrain_gnn_mode,
                                             "model.pth")
        molecule_model.from_pretrained(pretrained_model_path)
        molecule_model = molecule_model.to(device)

        kwargs["molecule_model"] = molecule_model
        molecule_dim = args.gnn_emb_dim

    elif molecule_type == "Motif":
        if args.dataset == "PubChemSTM":
            dataset = PubChemSTM_Datasets_GraphMotif(dataset_root, config.datafile)
        elif args.dataset == "PubChemSTM1K":
            dataset = PubChemSTM_SubDatasets_GraphMotif(dataset_root, datafile=config.datafile, size=1000)
        elif args.dataset == "PubChemSTM10K":
            dataset = PubChemSTM_SubDatasets_GraphMotif(dataset_root, datafile=config.datafile, size=10000)
        elif args.dataset == "PubChemSTM_Raw":
            dataset = PubChemSTM_Datasets_Raw_Graph(dataset_root)
        elif args.dataset == "PubChemSTM1K_Raw":
            dataset = PubChemSTM_SubDatasets_Raw_Graph(dataset_root, size=1000)
        elif args.dataset == "PubChemSTM10K_Raw":
            dataset = PubChemSTM_SubDatasets_Raw_Graph(dataset_root, size=10000)
        dataloader_class = pyg_DataLoader

        molecule_node_model = GNN(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
            JK=args.JK, drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type)
        molecule_model = GNN_motifpred(
            num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
            num_tasks=1, molecule_node_model=molecule_node_model)
        pretrained_model_path = os.path.join(args.dataspace_path, "pretrained_GraphMVP", args.pretrain_gnn_mode,
                                             "model.pth")
        molecule_model.from_pretrained(pretrained_model_path)
        if args.input_model_dir != "modelNone":
            input_model_path = os.path.join("checkpoints/{}".format(args.input_model_dir),
                                            "molecule_model_{}.pth".format(args.last_epoch))
            print("Loading from {}...".format(input_model_path))
            state_dict = torch.load(input_model_path, map_location="cpu")
            molecule_model.load_state_dict(state_dict)
        molecule_model = molecule_model.to(device)

        kwargs["molecule_model"] = molecule_model
        molecule_dim = args.gnn_emb_dim

    else:
        raise Exception
    a = dataset[0]
    dataloader = dataloader_class(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True  
    )

    if args.text_type == "SciBERT":
        pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'pretrained_SciBERT')
        text_tokenizer = AutoTokenizer.from_pretrained('../data/pretrained_SciBERT', )
        # text_model = AutoModel.from_pretrained('../data/pretrained_SciBERT').to(device)
        # text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder, force_download=True)
        # text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
        kwargs["text_tokenizer"] = text_tokenizer
        # kwargs["text_model"] = text_model
        text_dim = 768
    else:
        raise Exception

    # text2latent = nn.Linear(text_dim, args.SSL_emb_dim).to(device)
    mol2latent = nn.Linear(molecule_dim, args.SSL_emb_dim).to(device)
    frag_classifier = nn.Sequential(
        nn.Linear(args.SSL_emb_dim, args.SSL_emb_dim),  # projection
        nn.Linear(args.SSL_emb_dim, config.v_target_size)
    ).to(device)
    frag_classifier2 = nn.Sequential(
        nn.Linear(2 * args.SSL_emb_dim, args.SSL_emb_dim),  # projection
        nn.Linear(args.SSL_emb_dim, config.v_target_size)
    ).to(device)

    if args.input_model_dir != "modelNone":

        input_model_path = os.path.join("checkpoints/{}".format(args.input_model_dir),
                                        "frag_model_{}.pth".format(args.last_epoch))
        print("Loading from {}...".format(input_model_path))
        if os.path.exists(input_model_path):
            state_dict = torch.load(input_model_path, map_location='cpu')
            frag_classifier.load_state_dict(state_dict)

    if args.input_model_dir != "modelNone":
        input_model_path = os.path.join("checkpoints/{}".format(args.input_model_dir),
                                        "mol2latent_model_{}.pth".format(args.last_epoch))
        print("Loading from {}...".format(input_model_path))
        state_dict = torch.load(input_model_path, map_location='cpu')
        mol2latent.load_state_dict(state_dict)

    model = BertForMultiModalPreTraining(config).to(device)
    if args.input_model_dir != "modelNone":
        input_model_path = os.path.join("checkpoints/{}".format(args.input_model_dir),
                                        "multi_model_{}.pth".format(args.last_epoch))
        print("Loading from {}...".format(input_model_path))
        loaded_state_dict = torch.load(input_model_path, map_location='cpu')
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in loaded_state_dict.items() if
                               k in model_state_dict and model_state_dict[k].size() == loaded_state_dict[k].size()}
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
    kwargs["model"] = model

    # 使用 accelerator 的随机种子设置
    set_seed(args.seed)

    if args.representation_frozen:
        print("Representation is fronzen during pretraining.")
        freeze_network(molecule_model)
        model_param_group = [
            {"params": mol2latent.parameters(), "lr": config.mol_lr * config.mol_lr_scale},
        ]
    else:
        model_param_group = [
            {"params": molecule_model.parameters(), "lr": config.mol_lr},
            {"params": mol2latent.parameters(), "lr": config.mol_lr * config.mol_lr_scale},
            {"params": model.parameters(), "lr": config.multi_lr},
            {"params": frag_classifier.parameters(), "lr": config.frag_lr},
        ]
    optimizer = optim.Adam(model_param_group, weight_decay=args.decay)
    optimal_loss = 1e10

    plt.ion()
    fig, ax = plt.subplots()
    next_losses = []
    motif_losses = []
    lm_losses = []
    frag_losses = []
    frag_losses2 = []

    # 将模型和优化器传给 accelerator
    model, molecule_model, mol2latent, frag_classifier, optimizer = accelerator.prepare(
        model, molecule_model, mol2latent, frag_classifier, optimizer
    )

    # 准备数据加载器
    dataloader = accelerator.prepare_data_loader(dataloader)

    if args.input_model_dir != "modelNone":
        if isinstance(args.last_epoch, int):
            last = int(args.last_epoch) + 1
        else:
            last = 1
    else:
        last = 1
    try:
        for e in range(last, args.epochs + 1):
            try:
                print("Epoch {}".format(e))
                start = time.time()
                train(e, dataloader, wandb_run, **kwargs)
                end = time.time()
                print(f"Epoch {e} completed in {end - start} seconds")
            except Exception as e:
                print(f"Error in epoch {e}: {str(e)}")
                raise e
    except Exception as e:
        print(f"Global error: {str(e)}")
        raise e

    save_model(save_best=False)


