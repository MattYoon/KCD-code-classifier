import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gensim.models import FastText
from pytorch_lightning import LightningModule
import numpy as np
import re
import random
import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaModel, BertTokenizer
from .Preprocessor import Preprocessor

def set_seeds(seed=42):
    # 랜덤 시드를 설정하여 매 코드를 실행할 때마다 동일한 결과를 얻게 합니다.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.benchmark = False

class Tokenizer:
    def __init__(self, bert_tokenizer, w2v):
        self.bert_tokenizer = bert_tokenizer
        self.w2v = w2v

    def embed(self, sent):
        tokens = sent.split()
        embeddings = []
        for token in tokens:
            try:
                embedding = self.w2v.wv.get_vector(token)
                embeddings.append(embedding)
            except IndexError:  # 특수기호 때문에 룩업이 안되는 예외처리
                splits = re.sub(r'[^a-zA-Z]', ' ', token).split()
                for split in splits:
                    try:
                        embedding = self.w2v.wv.get_vector(split)
                        embeddings.append(embedding)
                    except IndexError:  # 진짜 임베딩이 없는 단어인 경우
                        pass
        if not len(embeddings):
            embeddings.append(self.w2v.wv.get_vector(''))
        embeddings = np.array(embeddings)

        return np.mean(embeddings, axis=0)

    def __call__(self, x):
        ko, eng = x['ko'], x['eng']
        bert_tokens = self.bert_tokenizer(
            ko, max_length=200, truncation=True, padding=True, return_tensors='pt')
        med_embeddings = torch.tensor(self.embed(eng))

        return {'bert_tokens': bert_tokens, 'med_embeddings': torch.unsqueeze(med_embeddings, 0)}


class MedClassifier(LightningModule):
    def __init__(self):
        super().__init__()

        self.bert_tokenizer = BertTokenizer.from_pretrained('klue/roberta-large')
        print('Loading W2V, may take some time > 5 minutes')
        self.w2v = FastText.load('demo/model/FT_300/ft_oa_all_300d.bin')
        self.preprocessor = Preprocessor()
        self.tokenizer = Tokenizer(self.bert_tokenizer, self.w2v)

        self.config = RobertaModel.from_pretrained('klue/roberta-large').config
        self.label_encoder = LabelEncoder()
        df = pd.read_csv('demo/server/data_no_sparse2.csv')
        self.label_encoder.fit(df['진단코드'])

        self.bert = RobertaModel(self.config)
        self.fnn1 = nn.Sequential(
            nn.Linear(1324, 1200),
            nn.PReLU(),
            nn.Linear(1200, 956)
        )

    def forward(self, x):
        x = self.preprocessor(x)
        x = self.tokenizer(x)

        x_bert, x_med = x['bert_tokens'], x['med_embeddings']
        bert_out = self.bert(x_bert['input_ids'],
                                attention_mask=x_bert['attention_mask']).last_hidden_state[:, 0]

        sent_emb = torch.cat((bert_out, x_med), dim=1)
        y = self.fnn1(sent_emb)
        logit = F.log_softmax(y, dim=1)
        _, top_n_preds = torch.topk(logit, k=5, dim=1)
        return self.label_encoder.inverse_transform(top_n_preds[0]).tolist()

def infer(x):
    return model(x)

print('Loading Model')
set_seeds()
dirpath = 'demo/model/checkpoint/'
checkpoint_filename = [f for f in listdir(dirpath) if isfile(join(dirpath, f))][0]
model = MedClassifier.load_from_checkpoint(dirpath + checkpoint_filename)
print('Model Loaded')