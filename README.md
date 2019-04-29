# Hierarchical Attentional Hybrid Neural Networks for Document Classification

J. Abreu , L. Fred, D. Macêdo, C. Zanchettin, "[**Hierarchical Attentional Hybrid Neural Networks for Document Classification**](https://arxiv.org/abs/1901.06610)", Submitted to IJCNN on 15 Jan, 2019.


## Datasets:
| Dataset                | Classes | Documents | source |
|------------------------|:---------:|:-------:|:--------:|
| Yelp Review Polarity   |    5    |    1569264   |[link](https://www.kaggle.com/luisfredgs/hahnn-for-document-classification)|
| IMDb Movie Review      |    2    |    50000       | [link](https://www.kaggle.com/luisfredgs/hahnn-for-document-classification)|

To download datasets, install the kaggle tool:

``` pip install kaggle ``` 

then run follow command:

``` kaggle datasets download -d luisfredgs/hahnn-for-document-classification ```

``` kaggle datasets download -d luisfredgs/wiki-news-300d-1m-subword ```

Or provide your kaggle username and API Key on **hahnn-for-document-classification.ipynb**

## Requirements

* tensorflow 1.10
* Keras 2.x
* spacy 2.0
* gensim
* tqdm
* matplotlib

A GPU with CUDA support is required to run this code

