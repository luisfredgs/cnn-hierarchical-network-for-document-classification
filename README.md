# Hierarchical Attentional Hybrid Neural Networks for Document Classification

J. Abreu , L. Fred, D. Macêdo, C. Zanchettin, "[**Hierarchical Attentional Hybrid Neural Networks for Document Classification**](https://arxiv.org/abs/1901.06610)", Submitted to IJCNN on 15 Jan, 2019.


## Datasets:
| Dataset                | Classes | Documents | source |
|------------------------|:---------:|:-------:|:--------:|
| Yelp Review Polarity   |    5    |    1569264   |[link](https://www.kaggle.com/luisfredgs/in1164-deep-learning)|
| IMDb Movie Review      |    2    |    50000       | [link](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)|

To download datasets, install the kaggle tool:

``` pip install kaggle ``` 

then:

``` kaggle datasets download -d luisfredgs/in1164-deep-learning ```

put all data on ***input/*** folder

## Requirements

* tensorflow 1.10
* Keras 2.x
* gensim
* tqdm
* matplotlib

## Execution 
run ```python train.py```
