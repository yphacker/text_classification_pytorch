# Toxic Comment Classification Challenge
[竞赛链接](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
## 数据下载
[data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
## 评估标准
the score is the average of the individual AUCs of each predicted column
## kaggle score:
5折
|model|offline score|online score|note|
|:---:|:---:|:---:|:---:|
|ml|-1|0.97016|
|cnn|-1|0.96977|
|cnn+预训练|-1|0.97571|
|rnn|-1|0.95663|
|rnn+atten|-1|0.97542|
|rcnn|-1|0.97468|
|bert(bert-base-uncased)|0.9900|0.98563|0.9895,0.9912,0.9899,0.9888,0.9908|
|albert(albert-base-v2)|0.9720|-1|0.9686,0.9741,0.9714,0.9730,0.9727|
|xlmroberta(xlm-roberta-base)||||
|bart(bart-large-cnn)||||

单折
|model|offline score|note|
|:---:|:---:|:---:|:---:|
|ml|-1||
|cnn|-1||
|cnn+预训练|-1||
|rnn|-1||
|rnn+atten|-1||
|rcnn|-1||
|bert(bert-base-uncased)|-1||
|albert(albert-base-v2)|98.22|epoch=5就不再提升了|
|xlmroberta(xlm-roberta-base)||||
|bart(bart-large-cnn)||||


## 实验环境
Tesla P100
16G
cuda9  
python:3.6  
torch:1.2.0.dev20190722

## script
nohup python main.py -m='cnn' -b=256 -e=3 > nohup/cnn.out 2>&1 &  
nohup python main.py -m='bert' -b=32 -e=4 > nohup/bert.out 2>&1 &  
nohup python main.py -m='albert' -b=64 -e=8 -mode=2 > nohup/albert.out 2>&1 & 
nohup python main.py -m='albert' -b=64 -e=5 > nohup/albert.out 2>&1 &  
nohup python main.py -m='xlmroberta' -b=20 -e=2 > nohup/xlmroberta.out 2>&1 &  
python predict.py -m='bert'  

## 参考文章

## 参考代码
[1] [google-research/bert](https://github.com/google-research/bert)  
[2] [google-research/ALBERT](https://github.com/google-research/ALBERT)  
[3] [huggingface/transformers](https://github.com/huggingface/transformers)  
[4] [649453932/Bert-Chinese-Text-Classification-Pytorch](https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch)  