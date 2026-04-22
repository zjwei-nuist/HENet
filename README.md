# HENet: A Heterogeneous Encoding Network for General and Robust Adversarial Example Generation (IEEE TIFS 2026)

Pytorch implementation of the paper "HENet: A Heterogeneous Encoding Network for General and Robust Adversarial Example Generation" by Jiawei Zhang, Hao Wang, Hao Wu, Bin Li, Xiangyang Luo, Bin Ma, Jinwei Wang.

Published in IEEE Transactions on Information Forensics and Security (TIFS 2026).

## Requirements
You need to install the requirements before the training and testing.
You need to download the dataset and decompress the data to the corresponding path within the dataset folder. For example, the default path for Caltech 256 is ./HENet/dataset/caltech256/256_ObjectCategories/...

## Running
You can run eval_HENet.py or eval_JPEG.py for the evaluation of HENet. (If you download all the pre-training weights (including different target networks and HENets) that we provide on Hugging Face.)

If you want to train your own HENet and evaluate on other target networks or datasets, you need to train the target networks before performing attacks in undistorted or distorted scenarios.

Then, you can run the main_dense_121_vit_b.py within each sub-folder of 'distorted' and 'undistorted' for training HENet.

Due to the storage restrictions of GitHub, we are unable to provide the weight of our HENet in this repository now. However, the full release of our HENet can be found on Hugging Face (https://huggingface.co/zjwei-cqupt/HENet).

## Citation
If you find this repository helpful, you may cite:

@ARTICLE{11480207,
  author={Zhang, Jiawei and Wang, Hao and Wu, Hao and Li, Bin and Luo, Xiangyang and Ma, Bin and Wang, Jinwei},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={HENet: A Heterogeneous Encoding Network for General and Robust Adversarial Example Generation}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  keywords={Feeds;Antennas;Frequency modulation;Radio broadcasting;Filtering;Circuits and systems;Filters;Network architecture;LoRa;High frequency;Adversarial example;JPEG compression;transformers;convolutional neural networks},
  doi={10.1109/TIFS.2026.3683276}}
