# Toxic Comment Classification Challenge
[竞赛链接](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
## 数据下载
[data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
## 评估标准
roc_auc
## kaggle score:
|model|score|note|
|---|---|---|
|ml|0.97016|
|cnn|0.82168|
|cnn+预训练|0.84793|
|rnn|0.46636|
|rcnn|0.73367|
|bert(bert-base-uncased)|0.95262|
|albert(albert-base-v2)|0.94647|
batch size:8

## 实验环境
Tesla P100
16G
cuda9  
python:3.6  
torch:1.2.0.dev20190722


## 预训练模型
[huggingface/transformers](https://github.com/huggingface/transformers)



## 参考链接
[google-research/bert](https://github.com/google-research/bert)  
[google-research/ALBERT](https://github.com/google-research/ALBERT)  
[huggingface/transformers](https://github.com/huggingface/transformers)  
[649453932/Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)  