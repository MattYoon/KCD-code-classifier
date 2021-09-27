import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import FastText
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
import re
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.classification.accuracy import accuracy
from transformers import ElectraTokenizer, ElectraModel, RobertaModel, RobertaTokenizer, AutoTokenizer, BertTokenizer
from utils import top_n_accuracy

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 16
FILE_DIR = 'data_no_sparse.csv'

# --------------------------
# 텍스트 데이터를 정수형으로 바꾸고, train, valid, test로 split
# --------------------------


class Preprocessor:
    def __init__(self, bert_tokenizer, w2v, label_encoder, train_ratio=0.95, file_dir=FILE_DIR):
        self.bert_tokenizer = bert_tokenizer
        self.w2v = w2v
        self.label_encoder = label_encoder

        self.train_ratio = train_ratio
        self.file_dir = file_dir

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

    def tokenize(self, df):
        print('Running Tokenizers')

        bert_tokens = self.bert_tokenizer(
            df['주호소 및 현병력'].tolist(), max_length=200, truncation=True, padding=True, return_tensors='pt')

        med_embeddings = torch.tensor(
            df['주호소 및 현병력 영문'].apply(self.embed).tolist())

        vectors = {'bert_tokens': bert_tokens,
                   'med_embeddings': med_embeddings}
        labels = {}
        for column in df.iloc[:, 2:]:
            labels[column] = torch.tensor(df[column].tolist())

        vectors['labels'] = labels

        return vectors

    def __call__(self):
        file = os.path.join(self.file_dir)
        df = pd.read_csv(file)
        # df = df.iloc[:100]  # Testing
        df = df.iloc[:, 0:7]  # 진료의세부분야, 진료과 drop
        df = df.fillna('')

        class_count = df.agg(['nunique'])
        for column in df.iloc[:, 2:]:
            classes[column] = class_count[column][0]

        for column in df.iloc[:, 2:]:
            self.label_encoder.fit(df[column])
            df[column] = self.label_encoder.transform(
                df[column])  # Target 값들을 정수형으로 변경

        df_train, df_valid = train_test_split(
            df, test_size=1-self.train_ratio, shuffle=True, random_state=1)
        # df_valid, df_test = train_test_split(
        #     df_tmp, test_size=0.5, shuffle=True, random_state=1)

        return self.tokenize(df_train), self.tokenize(df_valid)


class MedDataset(Dataset):
    def __init__(self, data):
        self.bert_tokens = data['bert_tokens']
        self.med_embeddings = data['med_embeddings']
        self.labels = data['labels']

    def __len__(self):
        return self.med_embeddings.shape[0]

    def __getitem__(self, index):
        bert_token = {key: val[index] for key, val in self.bert_tokens.items()}
        label = {key: val[index] for key, val in self.labels.items()}
        return {'bert_token': bert_token, 'med_embedding': self.med_embeddings[index], 'label': label}


class MedClassifier(LightningModule):
    def __init__(self, train_data, valid_data, bert_config, pretrained_lm):
        super().__init__()
        self.train_set = train_data
        self.valid_set = valid_data

        self.bert = RobertaModel(bert_config)
        if pretrained_lm:
            self.bert = RobertaModel.from_pretrained(
                "klue/roberta-large")

        self.sent_emb_size = 1324

        self.fnn1 = self.fnn(0, classes['진단코드'], 1200)

    def fnn(self, input_size, output_size, hidden_size1):
        return nn.Sequential(
            nn.Linear(self.sent_emb_size + input_size, hidden_size1),
            nn.PReLU(),
            nn.Linear(hidden_size1, output_size)
        )

    def forward(self, x):
        x_bert, x_med = x['bert_token'], x['med_embedding']
        bert_out = self.bert(x_bert['input_ids'],
                             attention_mask=x_bert['attention_mask']).last_hidden_state[:, 0]  # CLS Token, (batch_size, hidden_size

        sent_emb = torch.cat((bert_out, x_med), dim=1)

        logit = {}
        x = self.fnn1(sent_emb)
        logit['진단코드'] = F.log_softmax(x, dim=1)
        return logit

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.nll_loss(logits[TARGET], batch['label'][TARGET])

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = F.nll_loss(logits[TARGET], batch['label'][TARGET])
        preds = torch.argmax(logits[TARGET], dim=1)
        acc = accuracy(preds, batch['label'][TARGET])
        top_3_acc = top_n_accuracy(logits[TARGET], batch['label'][TARGET])
        top_5_acc = top_n_accuracy(logits[TARGET], batch['label'][TARGET], n=5)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_3_acc", top_3_acc, prog_bar=True)
        self.log("val_5_acc", top_5_acc, prog_bar=True)
        return loss

    def setup(self, stage):
        if stage == 'fit':
            print('\n**********************************************************')
            print(
                f'Fitting on: {TARGET}, Target Class Size: {classes[TARGET]}')
            print('**********************************************************\n')
        if stage == 'validate':
            print('\n**********************************************************')
            print('validating')
            print('**********************************************************\n')

    def configure_optimizers(self):
        print(f'Setting Optimizer on target {TARGET}')
        optimizer = torch.optim.AdamW([
            {'params': self.bert.parameters(), 'lr': 2e-5},
            {'params': self.fnn1.parameters(), 'lr': 2e-3},
        ])
        scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=BATCH_SIZE, num_workers=32)


if __name__ == '__main__':

    classes = {  # 값은 Preprocessor에서 초기화
        '진단코드': 0,
        '진단분류': 0,
        '진단소분류': 0,
        '진단중분류': 0,
    }

    bert_tokenizer = BertTokenizer.from_pretrained("klue/roberta-large")
    bert_config = RobertaModel.from_pretrained(
        "klue/roberta-large").config
    print('Loading W2V (This may take some time.. > 5 minutes)')
    w2v = FastText.load('FT_300/ft_oa_all_300d.bin')
    label_encoder = LabelEncoder()

    preprocessor = Preprocessor(bert_tokenizer, w2v, label_encoder)
    train_data, valid_data = preprocessor()

    train_set = MedDataset(train_data)
    valid_set = MedDataset(valid_data)

    model = MedClassifier(train_data=train_set, valid_data=valid_set,
                          bert_config=bert_config, pretrained_lm=True)

    TARGET = '진단코드'

    checkpoint_dir = "checkpoint_final/"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        filename="{v_num:02d}-{epoch:02d}-{val_acc:.3f}-{val_3_acc:.3f}-{val_5_acc:.3f}",
        mode="min",
    )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        callbacks=[checkpoint_callback],
        max_epochs=15,
        progress_bar_refresh_rate=5,
        accumulate_grad_batches=2,
    )
    trainer.fit(model)
