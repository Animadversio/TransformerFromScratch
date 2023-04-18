import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#%% Multi head attention
embdim = 768
headcnt = 12
headdim = embdim // headcnt
assert headdim * headcnt == embdim
tokens = torch.randn(1, 5, embdim) # batch, tokens, embedding
Wq = torch.randn(embdim, headcnt * headdim) / math.sqrt(embdim) # heads packed in a single dim
Wk = torch.randn(embdim, headcnt * headdim) / math.sqrt(embdim) # heads packed in a single dim
Wv = torch.randn(embdim, headcnt * headdim) / math.sqrt(embdim) # heads packed in a single dim

batch, token_num, _ = tokens.shape
qis = torch.einsum("BSE,EH->BSH", tokens, Wq)
kis = torch.einsum("BTE,EH->BTH", tokens, Wk)
vis = torch.einsum("BTE,EH->BTH", tokens, Wv)
qis_mh = qis.view(batch, token_num, headcnt, headdim)
kis_mh = kis.view(batch, token_num, headcnt, headdim)
vis_mh = vis.view(batch, token_num, headcnt, headdim)

scoremat_mh = torch.einsum("BSCH,BTCH->BCST", qis_mh, kis_mh)  # batch x headcnt x seqlen (query) x seqlen (key)
attmat_mh = F.softmax(scoremat_mh / math.sqrt(headdim), dim=-1)
zis_mh = torch.einsum("BCST,BTCH->BSCH", attmat_mh, vis_mh)  # batch x seqlen (query) x headcnt x headdim
# zis_mh = torch.einsum("BSCT,BTCH->BSCH", attmat_mh, vis_mh)
zis = zis_mh.reshape(batch, token_num, headcnt * headdim)
#%%
mha = nn.MultiheadAttention(embdim, headcnt, batch_first=True,)
# print(mha.in_proj_weight.shape) # 3 * embdim x embdim
mha.in_proj_weight.data = torch.cat([Wq, Wk, Wv], dim=1).T
attn_out, attn_weights = mha(tokens, tokens, tokens, average_attn_weights=False,)
assert torch.allclose(attmat_mh, attn_weights, atol=1e-6, rtol=1e-6)
assert torch.allclose(attn_out, mha.out_proj(zis), atol=1e-6, rtol=1e-6)

#%%
plt.figure()
for head in range(headcnt):
    plt.subplot(3, 4, head + 1)
    plt.imshow(attmat_mh[0, head].detach().numpy())
    plt.title(f"head {head}")
plt.show()

#%% Causal attention mask
attn_mask = torch.ones(token_num,token_num,)
attn_mask = -1E4 * torch.triu(attn_mask,1)
attn_mask

#%%
scoremat_mh_msk = torch.einsum("BSCH,BTCH->BCST", qis_mh, kis_mh)  # batch x headcnt x seqlen (query) x seqlen (key)
scoremat_mh_msk += attn_mask  # add the attn mask to the scores before SoftMax normalization
attmat_mh_msk = F.softmax(scoremat_mh_msk / math.sqrt(headdim), dim=-1)
zis_mh_msk = torch.einsum("BCST,BTCH->BSCH", attmat_mh_msk, vis_mh)  # batch x seqlen (query) x headcnt x headdim
# zis_mh = torch.einsum("BSCT,BTCH->BSCH", attmat_mh, vis_mh)
zis_msk = zis_mh_msk.reshape(batch, token_num, headcnt * headdim)

#%%
attn_out_causal, attn_weights_causal = mha(tokens, tokens, tokens, average_attn_weights=False, attn_mask=attn_mask)
assert torch.allclose(attn_weights_causal, attmat_mh_msk, atol=1e-6, rtol=1e-6)
assert torch.allclose(attn_out_causal, mha.out_proj(zis_msk), atol=1e-6, rtol=1e-6)

#%%
class TransformerBlock_simple(nn.Module):

    def __init__(self, embdim, headcnt, *args, dropout=0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ln1 = nn.LayerNorm(embdim)
        self.ln2 = nn.LayerNorm(embdim)
        self.attn = nn.MultiheadAttention(embdim, headcnt, batch_first=True,)
        self.ffn = nn.Sequential(
            nn.Linear(embdim, 4 * embdim),
            nn.GELU(),
            nn.Linear(4 * embdim, embdim),
            nn.Dropout(dropout),
        )

    def forward(self, x, is_causal=True):
        batch, token_num, _ = x.shape
        if is_causal:
            attn_mask = torch.ones(token_num, token_num,)
            attn_mask = -1E4 * torch.triu(attn_mask,1)
        else:
            attn_mask = None

        residue = x
        x = self.ln1(x)
        attn_output, attn_weights = self.attn(x, x, x, attn_mask=attn_mask)  # first output is the output latent states
        x = residue + attn_output

        residue = x
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        output = residue + ffn_output
        return output

#%%
from transformers import GPT2Model, GPT2Tokenizer
from transformers.activations import NewGELUActivation
model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()
#%%
#%%
inputs = tokenizer("Hi, I have a cat, her name is", return_tensors="pt")
outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
#%%
token_strs = tokenizer.tokenize("Hi, I have a cat, her name is")
#%%
print("Shape of final output token vectors", outputs.last_hidden_state.shape)
# attention of each GPTBlock:
print("num of attention outputs", len(outputs.attentions))
# shape of each attention tensor: [batch, heads, token (source), token (target)]
print("shape of each attention tensor", outputs.attentions[-1].shape)
print("num of hidden states (input embed included.) ", len(outputs.hidden_states))
print("shape of each hidden states tensor", outputs.hidden_states[-1].shape) #[batch, token, hidden]
assert torch.allclose(outputs.hidden_states[-1], outputs.last_hidden_state)
#%%

plt.figure(figsize=(10, 10))
for head in range(12):
    plt.subplot(3, 4, head + 1)
    plt.imshow(outputs.attentions[-1][0, head, :, :].detach().numpy())
    plt.yticks(range(len(token_strs)), token_strs)
    plt.xticks(range(len(token_strs)), token_strs)
plt.show()


#%% GPT2 from scratch
embdim = 768
headcnt = 12
tfmblock = TransformerBlock_simple(embdim, headcnt)

#%%
model.h[0].attn.c_attn.weight.shape
tfmblock.attn.in_proj_weight.shape
tfmblock.ln1.weight.data = model.h[0].ln_1.weight
tfmblock.ln1.bias.data = model.h[0].ln_1.bias
tfmblock.ln2.weight.data = model.h[0].ln_2.weight
tfmblock.ln2.bias.data = model.h[0].ln_2.bias
tfmblock.attn.in_proj_weight.data = model.h[0].attn.c_attn.weight.T
tfmblock.attn.in_proj_bias.data = model.h[0].attn.c_attn.bias
tfmblock.attn.out_proj.weight.data = model.h[0].attn.c_proj.weight.T
tfmblock.attn.out_proj.bias.data = model.h[0].attn.c_proj.bias
tfmblock.ffn[0].weight.data = model.h[0].mlp.c_fc.weight.T
tfmblock.ffn[0].bias.data = model.h[0].mlp.c_fc.bias
tfmblock.ffn[1] = NewGELUActivation()  # mlp in GPT2 and BERT used a new GELU activation, using nn.GeLU() will cause a small error around 1E-3
tfmblock.ffn[2].weight.data = model.h[0].mlp.c_proj.weight.T
tfmblock.ffn[2].bias.data = model.h[0].mlp.c_proj.bias
#%%
def GPT2block_to_TransformerBlock_simple(tfmblock, gpt2block, ):
    """copy the weights from a GPT2 block to a TransformerBlock_simple"""
    tfmblock.ln1.weight.data = gpt2block.ln_1.weight
    tfmblock.ln1.bias.data = gpt2block.ln_1.bias
    tfmblock.ln2.weight.data = gpt2block.ln_2.weight
    tfmblock.ln2.bias.data = gpt2block.ln_2.bias
    tfmblock.attn.in_proj_weight.data = gpt2block.attn.c_attn.weight.T
    tfmblock.attn.in_proj_bias.data = gpt2block.attn.c_attn.bias
    tfmblock.attn.out_proj.weight.data = gpt2block.attn.c_proj.weight.T
    tfmblock.attn.out_proj.bias.data = gpt2block.attn.c_proj.bias
    tfmblock.ffn[0].weight.data = gpt2block.mlp.c_fc.weight.T
    tfmblock.ffn[0].bias.data = gpt2block.mlp.c_fc.bias
    tfmblock.ffn[1] = NewGELUActivation()
    # mlp in GPT2 and BERT used a new GELU activation, using nn.GeLU() will cause a small error around 1E-3
    tfmblock.ffn[2].weight.data = gpt2block.mlp.c_proj.weight.T
    tfmblock.ffn[2].bias.data = gpt2block.mlp.c_proj.bias
    return tfmblock


def test_TransformerBlock_simple_GPT(block):
    tfmblock = TransformerBlock_simple(768, 12)
    GPT2block_to_TransformerBlock_simple(tfmblock, block)
    tokens_embs = torch.randn(2, 5, 768)
    tfmblock_out = tfmblock(tokens_embs, is_causal=True)
    block_out, = block(tokens_embs)
    assert torch.allclose(tfmblock_out, block_out, atol=1e-5, rtol=1e-5)


GPT2block_to_TransformerBlock_simple(tfmblock, model.h[0])
tokens_embs = torch.randn(2, 5, 768)
#%%
tfmblock_out = tfmblock(tokens_embs, is_causal=True)
modelblock_out, = model.h[0](tokens_embs)
#%%
assert torch.allclose(tfmblock_out, modelblock_out, atol=1e-5, rtol=1e-5)


#%%
class GPT2Model_simple(nn.Module):

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 768)
        self.wpe = nn.Embedding(1024, 768)
        self.blocks = nn.ModuleList([TransformerBlock_simple(768, 12) for _ in range(12)])
        self.ln_f = nn.LayerNorm(768)

    def forward(self, input_ids, input_embeds=None, is_causal=True):
        embeds = self.wte(input_ids) if input_embeds is None else input_embeds
        embeds = embeds + self.wpe(torch.arange(embeds.shape[1], device=embeds.device))
        for block in self.blocks:
            embeds = block(embeds, is_causal=is_causal)
        return self.ln_f(embeds)


def GPT2Model_to_GPT2Model_simple(gpt2modelsimple: GPT2Model_simple, gpt2model: GPT2Model):
    """copy the weights from a GPT2 model to a GPT2Model_simple"""
    gpt2modelsimple.wte.weight.data = gpt2model.wte.weight
    gpt2modelsimple.wpe.weight.data = gpt2model.wpe.weight
    gpt2modelsimple.ln_f.weight.data = gpt2model.ln_f.weight
    gpt2modelsimple.ln_f.bias.data = gpt2model.ln_f.bias
    for i in range(12):
        GPT2block_to_TransformerBlock_simple(gpt2modelsimple.blocks[i], gpt2model.h[i])
    return gpt2modelsimple


def test_our_GPT2(model: GPT2Model, tokenizer: GPT2Tokenizer,
                  text: str = "I have a cat, her name is"):
    model_ours = GPT2Model_simple()
    GPT2Model_to_GPT2Model_simple(model_ours, model)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, )
    hidden_last_ours = model_ours(inputs['input_ids'])
    assert torch.allclose(outputs.last_hidden_state, hidden_last_ours, atol=1e-5, rtol=1e-5)
    return model_ours

model_ours = test_our_GPT2(model, tokenizer)