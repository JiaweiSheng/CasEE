# CasEE
Source code for ACL2021 finding paper: [*CasEE: A Joint Learning Framework with Cascade Decoding for Overlapping Event Extraction*](https://aclanthology.org/2021.findings-acl.14/).

Event extraction (EE) is a crucial information extraction task that aims to extract event information in texts. This work studies the realistic event overlapping problem, where a word may serve as triggers with several types or arguments with different roles. To tackle the above issues, this work proposes a joint learning framework CasEE with cascade decoding for overlapping event extraction. Particularly, CasEE sequentially performs type detection, trigger extraction and argument extraction, where the overlapped targets are extracted separately conditioned on the specific former prediction. All the subtasks are jointly learned in a framework to capture dependencies among the subtasks. The evaluation demonstrates that CasEE achieves significant improvements on overlapping event extraction over previous competitive methods.



# Requirements

We conduct our experiments on the following environments:

```
python 3.6
CUDA: 9.0
GPU: Tesla T4
pytorch == 1.1.0
transformers == 4.9.1
```

# Datasets

Since ACE 2005 dataset has few overlapping problem, we adopt Chinese Financial Event Extraction dataset as our evaluation dataset.
The original dataset can be accessed at [this repo](https://github.com/TimeBurningFish/FewFC).
Here we re-split train/dev/test data since the original literature has different experimental settings.
Note that the re-splited data is avaliable at ``/dataset/FewFC/data``, and we adjust data format for simplicity of data loader.
To run the code on other dataset, you could also adjust the data as the data format presented.

# How to run

To run the code, you could sequentially run the code as following steps:

1. Data preprocessing: Generate cascading sampled data for training, achieving the cascading learning strategy of the framework (which has been generated at ``/dataset/FewFC/cascading_sampled``)：

```
python pre_cascading.py
```


2. Train/Dev/Test the model: Run as follows to train/dev/test the model:

```
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --output_model_path ./models_save/model.bin --do_train True --do_eval True --do_test True > logs/model.log &
```

The hyper-parameters are recorded in ``/utils/params.py``. 
We adopt ``bert-base-chinese`` as our pretrained language model. For extention, you could also try further hyper-parameters for even better performance.

# Citation

If you find this code useful, please cite our work:

```
@inproceedings{Sheng2021:CasEE,
    title = "{C}as{EE}: {A} Joint Learning Framework with Cascade Decoding for Overlapping Event Extraction",
    author = "Sheng, Jiawei and
      Guo, Shu and
      Yu, Bowen and
      Li, Qian and
      Hei, Yiming and
      Wang, Lihong and
      Liu, Tingwen and
      Xu, Hongbo",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.14",
    doi = "10.18653/v1/2021.findings-acl.14",
    pages = "164--174",
}
```

