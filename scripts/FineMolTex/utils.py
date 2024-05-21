import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# This is for BERT
def padarray(A, size, value=0):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant', constant_values = value)


# This is for BERT
def preprocess_each_sentence(sentence, tokenizer, max_seq_len, mask=None):
    text_input = tokenizer(
        sentence, truncation=True, max_length=max_seq_len,
        padding='max_length', return_tensors='np')
    
    input_ids = text_input['input_ids'].squeeze()
    if mask != None:
        # 获取input_ids，并找出非填充部分的indices
        # input_ids = text_input["input_ids"]
        non_padded_indices = (input_ids != tokenizer.pad_token_id).nonzero()

        # 随机选择一些非填充的词进行mask，例如15%的非填充词
        mask_arr = (torch.rand(input_ids[non_padded_indices].shape) < mask) \
                   * (input_ids[non_padded_indices] != 102) \
                   * (input_ids[non_padded_indices] != 103)
        # num_to_mask = int(len(non_padded_indices) * 0.15)
        # mask_indices = non_padded_indices[torch.randperm(len(non_padded_indices))[:num_to_mask]]
        expanded_vector = torch.zeros(max_seq_len).bool()
        original_length = mask_arr.size(0)
        expanded_vector[:original_length] = mask_arr.bool()
        # 将选中的词替换为特殊的[MASK]标记的token ID
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
        mask_labels = input_ids[expanded_vector]
        input_ids[expanded_vector] = mask_token_id

        attention_mask = text_input['attention_mask'].squeeze()

        sentence_tokens_ids = padarray(input_ids, max_seq_len)
        sentence_masks = padarray(attention_mask, max_seq_len)
        return [sentence_tokens_ids, sentence_masks, expanded_vector, mask_labels]
    else:
        attention_mask = text_input['attention_mask'].squeeze()

        sentence_tokens_ids = padarray(input_ids, max_seq_len)
        sentence_masks = padarray(attention_mask, max_seq_len)
        return [sentence_tokens_ids, sentence_masks]


# This is for BERT
def prepare_text_tokens(device, description, tokenizer, max_seq_len, mask = None):
    B = len(description)
    if mask == None:
        tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len, mask=mask) for idx in
                          range(B)]
        tokens_ids = [o[0] for o in tokens_outputs]
        masks = [o[1] for o in tokens_outputs]

        tokens_ids = torch.Tensor(tokens_ids).long().to(device)
        masks = torch.Tensor(masks).to(device)
        return tokens_ids, masks,
    else:
        tokens_outputs = [preprocess_each_sentence(description[idx], tokenizer, max_seq_len, mask = mask) for idx in range(B)]
        tokens_ids = [o[0] for o in tokens_outputs]
        masks = [o[1] for o in tokens_outputs]
        masktokens = [o[2] for o in tokens_outputs]
        masklabels = [o[3] for o in tokens_outputs]
        masktokens = torch.stack(masktokens, dim=0)
        masklabels = np.concatenate(masklabels, axis=0)
        # masklabels = torch.stack(masklabels, dim=1)
        tokens_ids = torch.Tensor(tokens_ids).long().to(device)
        masks = torch.Tensor(masks).to(device)
        return tokens_ids, masks, masktokens, masklabels

    
def get_molecule_repr_MoleculeSTM(molecule_data, mol2latent=None, molecule_type="SMILES", MegaMolBART_wrapper=None, molecule_model=None, device=None):
    if molecule_type == "SMILES":
        embedding, pad_mask = MegaMolBART_wrapper.smileslist2embedding(molecule_data)  # [pad, B, d], [pad, B]
        molecule_repr = embedding[0, :, :]  # [B, d]
    elif molecule_type == "Motif":
        molecule_repr = molecule_model(molecule_data)
        if isinstance(molecule_repr, torch.Tensor):
            molecule_repr= mol2latent(molecule_repr)
        # elif isinstance(molecule_repr, tuple):
        #     molecule_repr = molecule_repr[0]
        else:
            for i in range(len(molecule_repr)):
                molecule_repr[i] = mol2latent(molecule_repr[i])
    else:
        molecule_repr, _ = molecule_model(molecule_data)
    
    if mol2latent is not None and molecule_type != "Motif":
        molecule_repr = molecule_repr.to(device)
        molecule_repr = mol2latent(molecule_repr)

    return molecule_repr


def get_molecule_repr_MoleculeSTM2(molecule_data, mol2latent=None, molecule_type="SMILES", MegaMolBART_wrapper=None,
                                  molecule_model=None, device=None, model = None, args=None):
    if molecule_type == "SMILES":
        embedding, pad_mask = MegaMolBART_wrapper.smileslist2embedding(molecule_data)  # [pad, B, d], [pad, B]
        molecule_repr = embedding[0, :, :]  # [B, d]
    elif molecule_type == "Motif":
        molecule_repr = molecule_model(molecule_data)
        if isinstance(molecule_repr, torch.Tensor):
            molecule_repr = mol2latent(molecule_repr)
        else:
            for i in range(len(molecule_repr)):
                molecule_repr[i] = mol2latent(molecule_repr[i])
    else:
        molecule_repr, _ = molecule_model(molecule_data)
    if mol2latent is not None and molecule_type != "Motif":
        molecule_repr = molecule_repr.to(device)
        molecule_repr = mol2latent(molecule_repr)
    if isinstance(molecule_repr, torch.Tensor):
        m_attention_mask = torch.full((len(molecule_data.text), args.max_seq_len), float(0)).to(args.device)
        padded_tensor = torch.zeros((len(molecule_data.text), args.max_seq_len, args.SSL_emb_dim)).to(args.device)
        m_position_embedding = torch.zeros((len(molecule_data.text), args.max_seq_len, 1)).to(args.device)
        frag_pos = []
        for i in range(len(molecule_data)):
            indices = (molecule_data.frag_batch == i).nonzero(as_tuple=False).squeeze()

            if len(indices.size()) == 0:
                size = 1
            else:
                size = indices.size()[0]
            frag_pos.extend([i * args.max_seq_len + j for j in range(1, 1 + size)])
            if size > (args.max_seq_len - 1):
                padded_tensor[i, 1:args.max_seq_len, :] = molecule_repr[:(args.max_seq_len - 1)]
                m_attention_mask[i, :args.max_seq_len] = 1
            else:
                padded_tensor[i, 1:size + 1, :] = molecule_repr[indices]
                m_attention_mask[i, : size + 1] = 1
        frag_pos = torch.tensor(frag_pos).to(device)
    else:
        m_attention_mask = torch.full((len(molecule_data), args.max_seq_len), float(0)).to(args.device)
        padded_tensor = torch.zeros((len(molecule_data), args.max_seq_len, args.SSL_emb_dim)).to(args.device)
        m_position_embedding = torch.zeros((len(molecule_data), args.max_seq_len, 1)).to(args.device)
        num = 0
        for item in molecule_repr:
            if item.size(0) > (args.max_seq_len - 1):
                padded_tensor[num, 1:args.max_seq_len, :] = item[:(args.max_seq_len - 1)]
                m_attention_mask[num, :args.max_seq_len] = 1
                # attention_mask[]
            else:
                padded_tensor[num, 1:item.size(0) + 1, :] = item
                m_position_embedding[num, 1:item.size(0) + 1, :] = torch.tensor(molecule_data.positions[num]).unsqueeze(
                    -1)
                m_attention_mask[num, :item.size(0) + 1] = 1

            num = num + 1
    m_repr = padded_tensor
    d_attention_mask = torch.zeros_like(m_attention_mask)
    description_tokens_ids = torch.zeros_like(m_attention_mask).long()
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

    return pooled_output_m


def freeze_network(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def get_num_task_and_type(dataset):
    if dataset in ["esol", "freesolv", "lipophilicity"]:
        return 1, "regression"
    elif dataset in ["hiv", "bace", "bbbp"]:
        return 1, "classification"
    elif dataset == "tox21":
        return 12, "classification"
    elif dataset == "pcba":
        return 92, "classification"
    elif dataset == "muv":
        return 17, "classification"
    elif dataset == "toxcast":
        return 617, "classification"
    elif dataset == "sider":
        return 27, "classification"
    elif dataset == "clintox":
        return 2, "classification"
    raise ValueError("Invalid dataset name.")


def do_CL_eval(X, Y, neg_Y, args):
    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1) # B, 1, d

    Y = Y.unsqueeze(0)
    Y = torch.cat([Y, neg_Y], dim=0) # T, B, d
    Y = Y.transpose(0, 1)  # B, T, d
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze().unsqueeze(0) # B*T
    B = X.size()[0]
    labels = torch.zeros(B).long().to(logits.device)  # B*1

    criterion = nn.CrossEntropyLoss()

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    confidence = logits
    CL_conf = confidence.max(dim=1)[0]
    CL_conf = CL_conf.cpu().detach().numpy()

    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    return CL_loss, CL_conf, CL_acc


