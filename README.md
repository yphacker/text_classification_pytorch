# Toxic Comment Classification Challenge
[竞赛链接](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
## 数据下载
[data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
## 评估标准
the score is the average of the individual AUCs of each predicted column
## kaggle score:
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|ml|-1|0.97016|
|cnn|-1|0.96977|
|cnn+预训练|-1|0.97571|
|rnn|-1|0.95663|
|rnn+atten|-1|0.97542|
|rcnn|-1|0.97468|
|bert(bert-base-uncased)|0.9900|0.98563|0.9895,0.9912,0.9899,0.9888,0.9908|
|albert(albert-base-v2)|-1|0.94647|
|xlmroberta(xlm-roberta-base)||||

## 实验环境
Tesla P100
16G
cuda9  
python:3.6  
torch:1.2.0.dev20190722

# script
5-fold:  
nohup python main.py -m='bert' -b=32 -e=8 > nohup/bert.out 2>&1 &
nohup python main.py -m='albert' -b=64 -e=8 > nohup/albert.out 2>&1 &
python predict.py -m='bert'

单模：  
nohup python main.py -m='bert' -b=32 -e=8 -mode=2 > nohup/bert.out 2>&1 &

## 预训练模型
[huggingface/transformers](https://github.com/huggingface/transformers)

## 参考链接
[google-research/bert](https://github.com/google-research/bert)  
[google-research/ALBERT](https://github.com/google-research/ALBERT)  
[huggingface/transformers](https://github.com/huggingface/transformers)  
[649453932/Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)  