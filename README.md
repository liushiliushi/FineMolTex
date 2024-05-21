
The complete repository with dataset and checkpoints can be found in https://anonymous.4open.science/status/FineMolTex-2266.

## 1 Environment

Install environment:

```
conda env create -f environment.yaml
```

Then install packages:
```
# install metagron
git clone https://github.com/MolecularAI/MolBART.git --branch megatron-molbart-with-zinc
cd MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism
pip install .
cd ../../..

# install apex
git clone https://github.com/chao1224/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```

## 2 Pre-trained Checkpoints

### GNN and GraphMVP

The checkpoints of GraphMVP are on [Google Drive link](https://drive.google.com/drive/u/1/folders/1uPsBiQF3bfeCAXSDd4JfyXiTh-qxYfu6).

### SciBERT
SciBERT can be utilized as:
```
SciBERT_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
SciBERT_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
```

### MegaMolBART
MegaMolBART can be downloaded by running `downsload_MegaBolBART.sh`.
## 3 Scripts

### Pretraining

```
python pretrain.py --dataset PubChemSTM --input_model_dir modelNone --output_model_dir modeltmp --molecule_type Motif --device 2 --num_workers 8 --batch_size=16 --config_file=config/config.json --SSL_loss=InfoNCE
```

### Downstream: Zero-shot Structure-text Retrieval

**For DrugBank-Description**

```
python retrieval_Description_Pharmacodynamics.py --task=molecule_description_removed_PubChem --molecule_type=Motif --input_model_dir=model --last_epoch=10 --config_file=config/config.json --device=2 --output_model_dir=modeltmp
```

**DrugBank-Pharmacodynamics**

```
python retrieval_Description_Pharmacodynamics.py --task=molecule_pharmacodynamics_removed_PubChem --molecule_type=Motif --input_model_dir=model --last_epoch=10 --config_file=config/config.json --device=2 --output_model_dir=modeltmp
```

**DrugBank-ATC**

```
python retrieval_ATC.py --molecule_type=Motif --input_model_dir=model --last_epoch=10 --config_file=config/config.json --device=2 --output_model_dir=modeltmp
```

### Downstream: Zero-shot Text-based Molecule Editing

```
python generation_Alignment.py --input_model_dir=model --last_epoch=10 --MoleculeSTM_molecule_type=Motif --device=0 --output_model_dir=generation --config_file=config/config.json

python generation_Optimization.py --MoleculeSTM_molecule_type=Motif --MoleculeSTM_model_dir=model_generation --last_epoch=3 --config_file=config/config.json --last_epoch2=1 --epochs=80 --language_edit_model_dir=generation --input_description_id=101 --device=0 --output_model_dir=generation_101
```

### Downstream: Molecular Property Prediction

```
python property_prediction.py --dataset=bace --molecule_type=Motif
```



