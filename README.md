# MulSTE

This is the origin Pytorch implementation of MulSTE in the following paper: 

[KDD' 24] <u>*MulSTE: A Multi-view Spatio-temporal Learning Framework with Heterogeneous Event Fusion for Demand-supply Prediction*</u>

## Requirements

- Python 3.8
- matplotlib == 3.1.1
- numpy == 1.19.4
- pandas  == 1.3.3
- torch == 1.9.0
- transformers == 4.11.3
- tqdm == 4.62.3
- lxml == 4.6.3

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Folder Structure

```latex
└── MulSTE
    ├── config                 
    │   |──  data.conf          
    │   |──  data_zz.conf                          
    ├── lib         
    │   |──  figure_plot.py 
    │   |──  utils.py
    │   |──  utils_fix_normalization.py
    │   |──  utils_fix_normalization_zz.py                       
    ├── model
    │   |──  fine_tuned_bert
    │   |──  pre_trained_bert
    │   |──  MulSTE_model.py
    │   |──  MulSTE_model_zz.py
    ├── MulSTE_train.py
    ├── MulSTE_train_zz.py 
    ├── README.md
    └── requirements.txt
```

## Reproducibility

1. **Acquire Pre-trained BERT:** Download `pytorch_model.bin` from [Hugging Face](https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main) and put it into `model\pre_trained_bert\chinese-macbert-base` folder.
2. **Fine-tune BERT:** Run `Model_Fine_Tuning.ipynb` in `model\fine_tuned_bert` folder to fine-tune the Pre-trained BERT, and put the output `fine-tuned-bert.model` into `model\fine_tuned_bert` folder.
3. **Demand-supply Prediction:** Then, all training and inference processes can be run at once, as follows.

Shanghai:

```bash
python MulSTE_train.py
```

Zhengzhou: 

```bash
python MulSTE_train_zz.py
```

