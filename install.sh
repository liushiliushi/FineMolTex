#!/bin/bash
#conda create -n STM4 python=3.8
#conda activate STM4

# conda install -y -c rdkit rdkit=2020.09.1.0
pip install rdkit
#conda install -y -c conda-forge -c pytorch pytorch=1.9.1
conda install -y -c pyg -c conda-forge pyg==2.0.3
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl


pip install requests
pip install tqdm
pip install matplotlib
pip install spacy
pip install Levenshtein

# for SciBert
conda install -y boto3
pip install transformers

# for MoleculeNet
pip install ogb==1.2.0

# install pysmilesutils
python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git

pip install deepspeed

# install metagron
# pip install megatron-lm==1.1.5
# git clone https://github.com/MolecularAI/MolBART.git --branch megatron-molbart-with-zinc
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

cd MolBART/megatron_molbart/Megatron-LM-v1.1.5-3D_parallelism
pip install .
cd ../../..

# install apex
# wget https://github.com/NVIDIA/apex/archive/refs/tags/22.03.zip
# unzip 22.03.zip
# git clone https://github.com/chao1224/apex.git
