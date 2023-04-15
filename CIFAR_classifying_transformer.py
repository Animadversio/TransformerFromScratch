## Import transformers
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertModel, BertConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2Model
## import MNIST from torchvision package
from torchvision import datasets, transforms
## import torch
import os
from os.path import join
from tqdm import tqdm, trange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
#%%
from torchvision.datasets import MNIST, CIFAR10
dataset = CIFAR10(root='/home/binxu/Datasets', train=True, download=True, transform=
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]))
# augmentations are super important for CNN trainings, or it will overfit very fast without achieving good generalization accuracy
val_dataset = CIFAR10(root='/home/binxu/Datasets', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]))
#%%
from transformers import BertModel, BertTokenizer, BertConfig
config = BertConfig(hidden_size=256, intermediate_size=1024, num_hidden_layers=12,
                    num_attention_heads=8, max_position_embeddings=256,
                    vocab_size=100, bos_token_id=101, eos_token_id=102,
                    cls_token_id=103, )
model = BertModel(config).cuda()
patch_embed = nn.Conv2d(3, config.hidden_size, kernel_size=4, stride=4).cuda()
CLS_token = nn.Parameter(torch.randn(1, 1, config.hidden_size, device="cuda") / math.sqrt(config.hidden_size))
readout = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                        nn.GELU(),
                        nn.Linear(config.hidden_size, 10)
                        ).cuda()
for module in [patch_embed, readout, model, CLS_token]:
    module.cuda()

optimizer = AdamW([*model.parameters(),
                   *patch_embed.parameters(),
                   *readout.parameters(),
                   CLS_token], lr=5e-4)
#%%
batch_size = 128 # 96
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model.train()
for epoch in trange(50, leave=False):
    pbar = tqdm(train_loader, leave=False)
    for i, (imgs, labels) in enumerate(pbar):
        patch_embs = patch_embed(imgs.cuda())
        patch_embs = patch_embs.flatten(2).permute(0, 2, 1)  # (batch_size, HW, hidden)
        # print(patch_embs.shape)
        input_embs = torch.cat([CLS_token.expand(imgs.shape[0], 1, -1), patch_embs], dim=1)
        # print(input_embs.shape)
        output = model(inputs_embeds=input_embs)
        logit = readout(output.last_hidden_state[:, 0, :])
        loss = F.cross_entropy(logit, labels.cuda())
        # print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"loss: {loss.item():.4f}")

    # test on validation set
    model.eval()
    correct_cnt = 0
    total_loss = 0
    for i, (imgs, labels) in enumerate(val_loader):
        patch_embs = patch_embed(imgs.cuda())
        patch_embs = patch_embs.flatten(2).permute(0, 2, 1)  # (batch_size, HW, hidden)
        input_embs = torch.cat([CLS_token.expand(imgs.shape[0], 1, -1), patch_embs], dim=1)
        output = model(inputs_embeds=input_embs)
        logit = readout(output.last_hidden_state[:, 0, :])
        loss = F.cross_entropy(logit, labels.cuda())
        total_loss += loss.item() * imgs.shape[0]
        correct_cnt += (logit.argmax(dim=1) == labels.cuda()).sum().item()

    print(f"val loss: {total_loss / len(val_dataset):.4f}, val acc: {correct_cnt / len(val_dataset):.4f}")

# over fitting, validation accuracy
# patch size: 4
# loss: 0.2320: 100%|███████████████████████████| 521/521 [00:36<00:00, 14.45it/s]
# val loss: 1.3631, val acc: 0.6644
# 100%|███████████████████████████████████████████| 25/25 [16:14<00:00, 38.97s/it]

# over fitting, validation accuracy
# patch size: 8
# loss: 0.3418: 100%|██████████████████████████▉| 520/521 [00:36<00:00, 14.23it/s]
# val loss: 1.7761, val acc: 0.6016
#  96%|█████████████████████████████████████████▎ | 24/25 [15:06<00:38, 38.13s/it]

# with data augmentation, patch size: 4
# batch_size = 128, lr=5e-4 (AdamW)
# loss: 0.1783: 100%|███████████████████████████| 391/391 [00:40<00:00,  9.71it/s]
# val loss: 0.5719, val acc: 0.8309
#  86%|████████████████████████████████████▉      | 43/50 [29:29<05:07, 43.88s/it]

# loss: 0.1398: 100%|██████████████████████████▉| 390/391 [00:37<00:00, 10.69it/s]
# val loss: 0.6580, val acc: 0.8181
#%%
# Visual Transformers. Despite some previous work in which attention is used inside the convolutional layers of a CNN [57, 26], the first fully-transformer architectures for vision are iGPT [8] and ViT [17]. The former is trained using a "masked-pixel" self-supervised approach, similar in spirit to the common masked-word task used, for instance, in BERT [15] and in GPT [45] (see below). On the other hand, ViT is trained in a supervised way, using a special "class token" and a classification head attached to the final embedding of this token. Both methods are computationally expensive and, despite their very good results when trained on huge datasets, they underperform ResNet architectures when trained from scratch using only ImageNet-1K [17, 8]. VideoBERT [51] is conceptually similar to iGPT, but, rather than using pixels as tokens, each frame of a video is holistically represented by a feature vector, which is quantized using an off-the-shelf pretrained video classification model. DeiT [53] trains ViT using distillation information provided by a pretrained CNN
#%%
# https://openreview.net/pdf?id=SCN8UaetXx
from torchvision.models import resnet18
model_cnn = resnet18(pretrained=False)
model_cnn.fc = nn.Linear(512, 10)
model_cnn.cuda()
optimizer = AdamW(model_cnn.parameters(), lr=10e-4)
#%%
batch_size = 512
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
for epoch in trange(50, leave=False):
    model_cnn.train()
    pbar = tqdm(train_loader, leave=False)
    for i, (imgs, labels) in enumerate(pbar):
        output = model_cnn(imgs.cuda())
        loss = F.cross_entropy(output, labels.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"loss: {loss.item():.4f}")

    # test on validation set
    model_cnn.eval()
    correct_cnt = 0
    total_loss = 0
    for i, (imgs, labels) in enumerate(val_loader):
        output = model_cnn(imgs.cuda())
        loss = F.cross_entropy(output, labels.cuda())
        total_loss += loss.item() * imgs.shape[0]
        correct_cnt += (output.argmax(dim=1) == labels.cuda()).sum().item()

    print(f"val loss: {total_loss / len(val_dataset):.4f}, val acc: {correct_cnt / len(val_dataset):.4f}")

# batch size: 96, lr 1E-4
# loss: 0.1894: 100%|██████████████████████████▉| 519/521 [00:13<00:00, 37.36it/s]
#  96%|█████████████████████████████████████████▎ | 24/25 [06:03<00:15, 15.17s/it]
# val loss: 1.9748, val acc: 0.6292

# batch size: 256, lr 10E-4
#  96%|█████████████████████████████████████████▎ | 24/25 [03:41<00:09,  9.04s/it]
# loss: 0.0179:  99%|██████████████████████████▊| 195/196 [00:07<00:00, 24.92it/s]
# val loss: 1.6367, val acc: 0.7034

# batch size: 512, lr 10E-4 with data augmentation
# batch size: 512, lr 10E-4 with data augmentation
# loss: 0.4298: 100%|█████████████████████████████| 98/98 [00:16<00:00,  6.22it/s]
#  val loss: 0.6143, val acc: 0.8060
#  56%|████████████████████████                   | 28/50 [08:37<06:40, 18.21s/it]
# loss: 0.2113: 100%|█████████████████████████████| 98/98 [00:16<00:00,  6.28it/s]
# val loss: 0.6069, val acc: 0.8352
#  98%|██████████████████████████████████████████▏| 49/50 [14:58<00:18, 18.13s/it]


#%%
"""
CNN is an architecture with strong inductive bias. 
It will learn the data faster than transformer. But it's also faster to overfit. 
When proper data augmentation is applied, CNN will outperform transformer.

Transformer is an architecture with weak inductive bias
It will learn the data slower / needs more compute than CNN.   
"""
torch.save(model_cnn.state_dict(),"cnn_resnet18.pth")
#%%
torch.save(model.state_dict(),"bert.pth")