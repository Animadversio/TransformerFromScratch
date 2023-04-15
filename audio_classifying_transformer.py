import math
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2Model
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.nn.utils.rnn import pad_sequence
#%%
from torchfsdd import TorchFSDDGenerator, TrimSilence
from torchaudio.transforms import MFCC
from torchvision.transforms import Compose

# Create a transformation pipeline to apply to the recordings
transforms = Compose([
    TrimSilence(threshold=1e-6),
    MFCC(sample_rate=8e3, n_mfcc=64)
])

# Fetch the latest version of FSDD and initialize a generator with those files
fsdd = TorchFSDDGenerator(version='master', transforms=transforms,
                          ) #path="/home/binxu/Datasets"
#%%
# Create a Torch dataset for the entire dataset from the generator
full_set = fsdd.full()
# Create two Torch datasets for a train-test split from the generator
train_set, test_set = fsdd.train_test_split(test_size=0.1)
# Create three Torch datasets for a train-validation-test split from the generator
train_set, val_set, test_set = fsdd.train_val_test_split(test_size=0.15, val_size=0.15)
#%%
plt.figure()
plt.imshow(np.log(np.abs(train_set[100][0])))
plt.show()

#%%
def collate_fn(batch):
    # batch is a list of tuples, where each tuple is (audio_tensor, label_scalar)
    audios = []
    labels = []
    for audio, label in batch:
        audios.append(audio.T)  # time, freq features
        labels.append(label)
    # pad audio tensors to ensure they have the same length
    audios = pad_sequence(audios, batch_first=True, padding_value=0)
    # convert the labels list to a tensor
    labels = torch.tensor(labels)
    return audios, labels


#%%
# audio_tsrs, labels = next(iter(dataloaders))

#%%
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2Model
config = GPT2Config(n_embd=128, n_layer=12, n_head=16, n_positions=256,
                    vocab_size=100, bos_token_id=101, eos_token_id=102,
                    cls_token_id=103, )
MF_emb = nn.Linear(64, config.n_embd).cuda()
model = GPT2Model(config).cuda()
classifier_head = nn.Linear(config.n_embd, 10).cuda()
CLS_token = torch.randn(1, 1, config.n_embd).cuda() / math.sqrt(config.n_embd)
CLS_token = nn.Parameter(CLS_token)
optimizer = AdamW([*model.parameters(),
                  *MF_emb.parameters(),
                  *classifier_head.parameters(),
                   CLS_token], lr=1e-4)
#%%

dataloaders = DataLoader(train_set, batch_size=128, shuffle=True,
                         collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=256, shuffle=True,
                            collate_fn=collate_fn)
for epoch in trange(20):
    model.train()
    pbar = tqdm(dataloaders)
    for i, (audio, label) in enumerate(pbar):
        audio = audio.cuda()
        audio = MF_emb(audio)
        audio = torch.cat([audio, CLS_token.repeat(audio.shape[0], 1, 1)], dim=1)
        output = model(inputs_embeds=audio)
        last_hidden_state = output.last_hidden_state
        pooled_output = last_hidden_state[:, -1]
        logits = classifier_head(pooled_output)
        loss = F.cross_entropy(logits, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        test_corr_num = 0
        test_loss = 0
        for i, (audio, label) in enumerate(test_loader):
            audio = audio.cuda()
            audio = MF_emb(audio)
            audio = torch.cat([audio, CLS_token.repeat(audio.shape[0], 1, 1)], dim=1)
            output = model(inputs_embeds=audio)
            last_hidden_state = output.last_hidden_state
            pooled_output = last_hidden_state[:, -1]
            logits = classifier_head(pooled_output)
            loss = F.cross_entropy(logits, label.cuda())
            pbar.set_description(f"test loss: {loss.item():.4f}")
            test_corr_num += (logits.argmax(dim=1) == label.cuda()).float().sum()
            test_loss += loss.item()
        print(f"test acc: {test_corr_num / len(test_set):.4f}")
#%%
from transformers import BertModel, BertTokenizer, BertConfig
config = BertConfig(hidden_size=64, intermediate_size=256, num_hidden_layers=12,
                    num_attention_heads=8, max_position_embeddings=256,
                    vocab_size=100, bos_token_id=101, eos_token_id=102,
                    cls_token_id=103, )
model = BertModel(config).cuda()
# MF_emb = nn.Linear(64, config.hidden_size).cuda()
MF_emb = nn.Sequential(nn.Conv1d(64, config.hidden_size, 3, 1, 1),
                       nn.ReLU(),
                       nn.Conv1d(config.hidden_size, config.hidden_size, 3, 1, 1),
                       ).cuda()
classifier_head = nn.Linear(config.hidden_size, 10).cuda()
CLS_token = torch.randn(1, 1, config.hidden_size).cuda() / math.sqrt(config.hidden_size)
CLS_token = nn.Parameter(CLS_token)
optimizer = AdamW([*model.parameters(),
                  *MF_emb.parameters(),
                  *classifier_head.parameters(),
                   CLS_token], lr=1e-4)
# https://datasets.activeloop.ai/docs/ml/datasets/free-spoken-digit-dataset-fsdd/
# https://github.com/adhishthite/sound-mnist
#%%
dataloaders = DataLoader(train_set, batch_size=128, shuffle=True,
                         collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=256, shuffle=True,
                            collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=256, shuffle=True,
                            collate_fn=collate_fn)
for epoch in trange(40):
    model.train()
    pbar = tqdm(dataloaders)
    for i, (audio, label) in enumerate(pbar):
        audio = audio.cuda()
        audio = MF_emb(audio.permute(0, 2, 1)).permute(0, 2, 1)
        audio = torch.cat([CLS_token.repeat(audio.shape[0], 1, 1), audio, ], dim=1)
        output = model(inputs_embeds=audio)
        last_hidden_state = output.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        logits = classifier_head(pooled_output)
        loss = F.cross_entropy(logits, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        val_corr_num = 0
        val_loss = 0
        for i, (audio, label) in enumerate(val_loader):
            audio = audio.cuda()
            audio = MF_emb(audio.permute(0, 2, 1)).permute(0, 2, 1)
            audio = torch.cat([CLS_token.repeat(audio.shape[0], 1, 1), audio, ], dim=1)
            output = model(inputs_embeds=audio)
            last_hidden_state = output.last_hidden_state
            pooled_output = last_hidden_state[:, 0]
            logits = classifier_head(pooled_output)
            loss = F.cross_entropy(logits, label.cuda())
            val_corr_num += (logits.argmax(dim=1) == label.cuda()).float().sum()
            val_loss += loss.item()
        print(f"val acc: {val_corr_num / len(val_set):.4f}")

        test_corr_num = 0
        test_loss = 0
        for i, (audio, label) in enumerate(test_loader):
            audio = audio.cuda()
            audio = MF_emb(audio.permute(0, 2, 1)).permute(0, 2, 1)
            audio = torch.cat([CLS_token.repeat(audio.shape[0], 1, 1), audio, ], dim=1)
            output = model(inputs_embeds=audio)
            last_hidden_state = output.last_hidden_state
            pooled_output = last_hidden_state[:, 0]
            logits = classifier_head(pooled_output)
            loss = F.cross_entropy(logits, label.cuda())
            test_corr_num += (logits.argmax(dim=1) == label.cuda()).float().sum()
            test_loss += loss.item()
        print(f"test acc: {test_corr_num / len(test_set):.4f}")
# loss: 0.0476: 100%|██████████| 17/17 [00:28<00:00,  1.66s/it]
# val acc: 0.9833
# test acc: 0.9714
# 100%|██████████| 40/40 [25:56<00:00, 38.92s/it]