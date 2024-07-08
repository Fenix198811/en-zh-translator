1. Overall Introduction:

This is a repo written during LLM machine learning self study. It use the transfomer with attention model to train English-Chinese pair data and do sentence translation from English to Chinese.


2. Repo Structure:

data - including all the English-Chinese sentence pair data

results-in-terminal - including the terminal logs during model training and testing

transfomer - including all the tranfomer with attention model source code

constants.py - model constant params 

utils.py - some commonly used functions

training.py - the train and evaluate functions

dataset.py - load and preprocess the data and train the model

inference.py - define the translate sentence function

translator.py - load the model params saved during model training and do the sentence translation


3. Usage:

Traing model by running command: python3 dataset.py

Translate English into Chinese by running command: python translator.py


4. Reference:

a) https://arxiv.org/abs/1706.03762

b) https://colab.research.google.com/github/jaygala24/pytorch-implementations/blob/master/Attention%20Is%20All%20You%20Need.ipynb

c) https://colab.research.google.com/github/satyajitghana/TSAI-DeepNLP-END2.0/blob/main/13_AIAYN_Recap/Attention_is_All_You_Need_Modern.ipynb

d) https://colab.research.google.com/drive/1gOyMHlz5HvYn_cCInX9zLOOe1qHeu91b