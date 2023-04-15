maestro_v3_midi_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
maestro_v3_meta_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
maestro_v3_json_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json"
# %%
# download and unzip the urls above
# !wget $maestro_v3_midi_url
# !wget $maestro_v3_meta_url
# !wget $maestro_v3_json_url
# !unzip maestro-v3.0.0-midi.zip
# %%
import os
from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.optim import AdamW, Adam
from transformers import AdamW, get_linear_schedule_with_warmup
import librosa.display
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from typing import Union
import pretty_midi
import pickle

# maestro_root = r"E:\Datasets\maestro-v3.0.0-midi\maestro-v3.0.0"
maestro_root = r"/home/binxu/Datasets/maestro-v3.0.0"


# visualize the midi file as piano roll
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))


# load the metadata
maestro_meta = pd.read_csv(join(maestro_root, "maestro-v3.0.0.csv"))
# %%
# https://notebook.community/craffel/pretty-midi/Tutorial
# Each note has a start time, end time, pitch, and velocity.
#
# Velocity / volume of a note: 1-127
# Pitch of a note: 0-127 C1-G9
# Duration of a note: seconds, float number
# Start time of a note: seconds, float number

# %%
# midi to note sequence
def midi2notes(midi_file: Union[str, pretty_midi.PrettyMIDI]):
    if isinstance(midi_file, str):
        midi_data = pretty_midi.PrettyMIDI(midi_file)
    else:
        midi_data = midi_file

    note_t_seq = []
    note_dt_seq = []
    note_duration_seq = []
    note_velo_seq = []
    note_pitch_seq = []
    prev_note_end = 0.0
    for i, note_sample in enumerate(midi_data.instruments[0].notes[1:]):
        note_dt_seq.append(note_sample.start - prev_note_end)
        note_t_seq.append(note_sample.start)
        note_duration_seq.append(note_sample.duration)
        note_velo_seq.append(note_sample.velocity)
        note_pitch_seq.append(note_sample.pitch)
        prev_note_end = note_sample.end

    note_seq = pd.DataFrame({"t": note_t_seq, "dt": note_dt_seq, "duration": note_duration_seq, "velo": note_velo_seq,
                             "pitch": note_pitch_seq})  # .to_csv("note_seq.csv", index=False)
    return note_seq


midi_file = join(maestro_root, maestro_meta["midi_filename"][0].replace("/", os.path.sep))
note_seq = midi2notes(midi_file)
# %%
class MusicScoreDataset(torch.utils.data.Dataset):

    def __init__(self, maestro_root, maestro_meta=None, ):
        self.maestro_root = maestro_root
        if maestro_meta is None:
            self.maestro_meta = pd.read_csv(join(maestro_root, "maestro-v3.0.0.csv"))
        else:
            self.maestro_meta = maestro_meta

        self.dataset = {}

    def load_dataset(self):
        for i in tqdm(range(len(self.maestro_meta))):
            row = self.maestro_meta.iloc[i]
            midi_path = join(self.maestro_root, row["midi_filename"].replace("/", os.path.sep))
            # load sample midi file
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            note_seq = midi2notes(midi_data)
            self.dataset[i] = (row[["canonical_composer", 'canonical_title', 'year', 'duration']], note_seq)

        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        meta, note_seq = self.dataset[idx]
        return meta, note_seq


dataset = MusicScoreDataset(maestro_root, maestro_meta)
# %%
dataset.dataset = pickle.load(open(join(maestro_root, "dataset.pkl"), "rb"))
# %%
# import pickle
# pickle.dump(dataset.dataset, open(join(maestro_root, "dataset.pkl"), "wb"))
# %%
meta, note_seq = dataset[0]
# %%
# %%
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from torch.utils.data import DataLoader
# loader = DataLoader(dataset, batch_size=128, collate_fn=batch_sampler, shuffle=True)
# next(iter(loader))
token_num = 128 # 128 pitch classes
BOS_id = token_num
EOS_id = token_num + 1
PAD_ID = EOS_id + 1
MASK_LABEL_ID = - 100

def notedf2tensor(note_df, device="cuda"):
    pitch_ids = torch.tensor(note_df.pitch.values).unsqueeze(0).long().to(device)
    velo = torch.tensor(note_df.velo.values).unsqueeze(0).float().to(device) / 128.0
    dur = torch.tensor(note_df.duration.values).unsqueeze(0).float().to(device)
    dt = torch.tensor(note_df.dt.values).unsqueeze(0).float().to(device)
    return pitch_ids, velo, dur, dt


def batch_sampler(data_source, batch_idxs, max_seq_len=1024):
    batch_pitch = []  # integer value
    batch_velo = []  # integer value
    batch_dt = []  # float value
    batch_duration = []  # float value
    label_batch = []
    if isinstance(batch_idxs, int):
        batch_idxs = range(batch_idxs)
    for i in batch_idxs:
        meta, note_seq = data_source[i]
        # sample note sequence with max_seq_len
        if len(note_seq) <= max_seq_len:
            note_seq_subsamp = note_seq
        else:
            seq_start = np.random.randint(0, len(note_seq) - max_seq_len)
            note_seq_subsamp = note_seq.iloc[seq_start:seq_start + max_seq_len]

        pitch, velo, dur, dt = notedf2tensor(note_seq_subsamp, device="cpu")
        batch_pitch.append(pitch[0])
        batch_velo.append(velo[0])
        batch_dt.append(dt[0])
        batch_duration.append(dur[0])
        # batch_pitch.append(torch.tensor(note_seq_subsamp["pitch"].values).long())
        # batch_velo.append(torch.tensor(note_seq_subsamp["velo"].values).long())
        # batch_dt.append(torch.tensor(note_seq_subsamp["dt"].values))
        # batch_duration.append(torch.tensor(note_seq_subsamp["duration"].values))

    batch_pitch = pad_sequence(batch_pitch, batch_first=True, padding_value=PAD_ID)
    batch_velo = pad_sequence(batch_velo, batch_first=True, padding_value=0.5)
    batch_dt = pad_sequence(batch_dt, batch_first=True, padding_value=0.0)
    batch_duration = pad_sequence(batch_duration, batch_first=True, padding_value=0.0)
    # label_batch = pad_sequence(label_batch, batch_first=True, padding_value=MASK_LABEL_ID)
    return batch_pitch, batch_velo, batch_dt, batch_duration


batch_pitch, batch_velo, batch_dt, batch_duration = batch_sampler(dataset, range(32), max_seq_len=512)

# %% Submodules for a Music Note Transformer
class NoteReadoutHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.velocity_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh(),
            nn.Linear(config.n_embd, 1),
        )
        self.duration_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh(),
            nn.Linear(config.n_embd, 1),
            nn.Softplus(), # duration should be positive
        )
        self.dt_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.Tanh(),
            nn.Linear(config.n_embd, 1),
        )

    def forward(self, hidden_states):
        velocity = self.velocity_head(hidden_states).squeeze(-1)
        duration = self.duration_head(hidden_states).squeeze(-1)
        dt = self.dt_head(hidden_states).squeeze(-1)
        return velocity, duration, dt


class ScalarEmbedding(nn.Module):
    def __init__(self, config, ):
        super().__init__()
        self.n_embd = config.n_embd
        self.weights = nn.Parameter(torch.randn(1, 1, self.n_embd) * np.sqrt(1 / self.n_embd))

    def forward(self, x):
        return self.weights * x[:, :, None]

# %%

def naive_greedy_decode(model, prompt_ids, prompt_velos, prompt_durations, prompt_dts, max_gen_tokens=500,
                        temperature=None):
    model.eval()
    readout.eval()
    dt_emb.eval()
    duration_emb.eval()
    velocity_emb.eval()
    pitch_embed = model.get_input_embeddings()(prompt_ids)
    velo_embed = velocity_emb(prompt_velos)
    dur_embed = duration_emb(prompt_durations)
    dt_embed = dt_emb(prompt_dts)
    input_embed = pitch_embed + velo_embed + dur_embed + dt_embed
    input_embed_full = input_embed.clone()
    generated_note_seq = []
    with torch.no_grad():
        for i in trange(max_gen_tokens):
            out = model(inputs_embeds=input_embed_full, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]
            last_logits = out.logits[:, -1, :]
            # pred_velos, pred_durations, pred_dts = readout(last_hidden)
            pred_velo, pred_duration, pred_dt = readout(last_hidden[:, -1, :])
            if temperature is None:
                # greedy decoding
                pitch_maxprob = torch.argmax(last_logits, dim=-1)
            else:
                # Sample from the distribution with temperature
                pitch_maxprob = torch.multinomial(F.softmax(last_logits / temperature, dim=-1),
                                                  num_samples=1).squeeze(1)
            input_embed_new = model.get_input_embeddings()(pitch_maxprob) + \
                              velocity_emb(pred_velo.unsqueeze(0)) + \
                              duration_emb(pred_duration.unsqueeze(0)) + \
                              dt_emb(pred_dt.unsqueeze(0))
            generated_note_seq.append((pitch_maxprob, pred_velo, pred_duration, pred_dt))
            input_embed_full = torch.concat((input_embed_full, input_embed_new), dim=1)

    pitch_gen = torch.stack([pitch for pitch, velo, dur, dt in generated_note_seq], dim=1)
    velo_gen = torch.stack([velo for pitch, velo, dur, dt in generated_note_seq], dim=1)
    dur_gen = torch.stack([dur for pitch, velo, dur, dt in generated_note_seq], dim=1)
    dt_gen = torch.stack([dt for pitch, velo, dur, dt in generated_note_seq], dim=1)
    pitch_gen = torch.concat((prompt_ids, pitch_gen), dim=1).cpu()
    velo_gen = torch.concat((prompt_velos, velo_gen), dim=1).cpu()  # this is 0, 1 normalized
    dur_gen = torch.concat((prompt_durations, dur_gen), dim=1).cpu()
    dt_gen = torch.concat((prompt_dts, dt_gen), dim=1).cpu()
    velo_gen = (velo_gen * 128).long()  # convert to integer
    return pitch_gen, velo_gen, dur_gen, dt_gen


def tensor2midi(pitch, velo, dur, dt, savedir, filename, instrument="Acoustic Grand Piano"):
    generated_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(instrument)
    piano = pretty_midi.Instrument(program=piano_program)
    cur_time = 0
    for i, pitch in enumerate(pitch):
        cur_time = cur_time + dt[i].item()
        note = pretty_midi.Note(velocity=min(127, max(0, velo[i].item())),
                                pitch=min(127, max(0, pitch.item())),
                                start=cur_time, end=cur_time + dur[i].item())
        cur_time = cur_time + dur[i].item()
        piano.notes.append(note)

    generated_midi.instruments.append(piano)
    generated_midi.write(join(savedir, filename))
    return generated_midi



# %%
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
saveroot = "/home/binxu/DL_Projects/Maestro-GPT" #"runs"

# config = GPT2Config(n_embd=128, n_layer=12, n_head=8, n_positions=512, n_ctx=128,
#                     vocab_size=132, bos_token_id=BOS_id, eos_token_id=EOS_id, )
# config = GPT2Config(n_embd=128, n_layer=18, n_head=8, n_positions=512, n_ctx=128,
#                     vocab_size=token_num + 4, bos_token_id=BOS_id, eos_token_id=EOS_id, )
config = GPT2Config(n_embd=128, n_layer=36, n_head=8, n_positions=256,
                    vocab_size=token_num + 4, bos_token_id=BOS_id, eos_token_id=EOS_id, )
model = GPT2LMHeadModel(config)
readout = NoteReadoutHeads(config)
dt_emb = ScalarEmbedding(config).cuda()
duration_emb = ScalarEmbedding(config).cuda()
velocity_emb = ScalarEmbedding(config).cuda()
model.cuda()
readout.cuda()
optimizer = AdamW([*model.parameters(),
                   *readout.parameters(),
                   *dt_emb.parameters(),
                   *duration_emb.parameters(),
                   *velocity_emb.parameters(),
                   ], lr=10e-4)
scheduler = get_linear_schedule_with_warmup(optimizer, 50, 3000)
# savedir = join(saveroot, "runs_L18")
# savedir = join(saveroot, "runs_L18_lrsched_nexttok")
# savedir = join(saveroot, "runs_L36_ctx256_lrsched_nexttok")
savedir = join(saveroot, "runs_L36_ctx256_lrsched_nexttok_fixeval")
os.makedirs(savedir, exist_ok=True)
os.makedirs(join(savedir, "ckpt"), exist_ok=True)
# %%
save_per_epoch = 10
synth_per_epoch = 10
batch_size = 24
max_seq_len = 256
writer = SummaryWriter(savedir)
for epoch in trange(0, 3000):
    for module in [model, readout, dt_emb, duration_emb, velocity_emb]:
        module.train()
    rand_idx_seq = np.random.permutation(len(dataset))
    for i, csr in enumerate(trange(0, len(dataset), batch_size)):
        batch_idxs = rand_idx_seq[csr:csr + batch_size]
        batch_pitch, batch_velo, batch_dt, batch_duration = batch_sampler(dataset,
                                                      batch_idxs, max_seq_len=max_seq_len)
        pitch_embed = model.transformer.wte(batch_pitch.cuda())
        note_embed = pitch_embed + velocity_emb(batch_velo.cuda()) + \
            dt_emb(batch_dt.cuda()) + duration_emb(batch_duration.cuda())  #

        # out = model(batch_pitch.cuda(), labels=batch_pitch.cuda(), output_hidden_states=True)
        out = model(inputs_embeds=note_embed, labels=batch_pitch.cuda(), output_hidden_states=True)
        loss = out.loss
        last_hidden = out.hidden_states[-1]  # (batch, seq_len, embd)
        velocity_pred, duration_pred, dt_pred = readout(last_hidden)
        # note this is wrong, need to predict the next note so shift by 1
        # loss_velo = F.mse_loss(velocity_pred, batch_velo.cuda())
        # loss_duration = F.mse_loss(duration_pred, batch_duration.cuda())
        # loss_dt = F.mse_loss(dt_pred, batch_dt.cuda())
        # this is correct prediction
        loss_velo = F.mse_loss(velocity_pred[:, :-1], batch_velo.cuda()[:, 1:])
        loss_duration = F.mse_loss(duration_pred[:, :-1], batch_duration.cuda()[:, 1:])
        loss_dt = F.mse_loss(dt_pred[:, :-1], batch_dt.cuda()[:, 1:])
        loss += loss_velo + loss_duration + loss_dt
        # compute additional loss based on last hidden state
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(
            f"epoch{epoch}-step{i:03d} {loss.item():.5f} vel {loss_velo.item():.5f} dur {loss_duration.item():.5f} dt {loss_dt.item():.5f}")
        writer.add_scalar("loss", loss.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("loss_velo", loss_velo.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("loss_duration", loss_duration.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("loss_dt", loss_dt.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch * len(dataset) // batch_size + i)
        writer.add_scalar("epoch", epoch, epoch * len(dataset) // batch_size + i)
        # writer.add_scalar("grad_norm", torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),)
    scheduler.step()
    if epoch % save_per_epoch == 0:
        torch.save(model.state_dict(), join(savedir, "ckpt", f"model_{epoch:03d}.pt"))
        torch.save(readout.state_dict(), join(savedir, "ckpt", f"readout_{epoch:03d}.pt"))
        torch.save(dt_emb.state_dict(), join(savedir, "ckpt", f"dt_emb_{epoch:03d}.pt"))
        torch.save(duration_emb.state_dict(), join(savedir, "ckpt", f"duration_emb_{epoch:03d}.pt"))
        torch.save(velocity_emb.state_dict(), join(savedir, "ckpt", f"velocity_emb_{epoch:03d}.pt"))

    if epoch % synth_per_epoch == 0:
        prompt_ids, prompt_velos, prompt_durations, prompt_dts = notedf2tensor(note_seq[:10], device="cuda")
        pitch_gen, velo_gen, dur_gen, dt_gen = naive_greedy_decode(model,
                       prompt_ids, prompt_velos, prompt_durations, prompt_dts,
                        temperature=.2, max_gen_tokens=max_seq_len - 12)
        tensor2midi(pitch_gen[0], velo_gen[0], dur_gen[0], dt_gen[0],
                    savedir, f"generated_midi_with_tempo_T02_{epoch:03d}.mid")
#%%

torch.save(readout.state_dict(), join(savedir, "ckpt", f"readout_{epoch:03d}.pt"))
torch.save(dt_emb.state_dict(), join(savedir, "ckpt", f"dt_emb_{epoch:03d}.pt"))
torch.save(duration_emb.state_dict(), join(savedir, "ckpt", f"duration_emb_{epoch:03d}.pt"))
torch.save(velocity_emb.state_dict(), join(savedir, "ckpt", f"velocity_emb_{epoch:03d}.pt"))
model.save_pretrained(join(savedir, "ckpt", f"model_{epoch:03d}.pt"))
config.save_pretrained(join(savedir, "ckpt","config.json"))
#%%
# sampling from the model
model.eval()
readout.eval()
dt_emb.eval()
duration_emb.eval()
velocity_emb.eval()
# %%
prompt_ids = torch.tensor(note_seq.pitch[:10]).unsqueeze(0).long().cuda()
answers = model.generate(prompt_ids, max_length=128, do_sample=True,
                top_k=0, top_p=0.90, num_return_sequences=3,
                bos_token_id=BOS_id, eos_token_id=EOS_id, pad_token_id=PAD_ID,
                 output_hidden_states=True, return_dict_in_generate=True)



#%% Generate sequence
prompt_ids = torch.tensor(note_seq.pitch[:10]).unsqueeze(0).long().cuda()
answers = model.generate(prompt_ids, max_length=100, do_sample=True,
                top_k=0, top_p=0.90, num_return_sequences=3,
                bos_token_id=BOS_id, eos_token_id=EOS_id, pad_token_id=PAD_ID,
                 output_hidden_states=True, return_dict_in_generate=True)
#%%
#%%
# generate sequence with readout and embedding

#%%


# prompt_ids = torch.tensor(note_seq.pitch[:12]).unsqueeze(0).long().cuda()
# prompt_velos = torch.tensor(note_seq.velo[:12]).unsqueeze(0).cuda().float() / 128.0
# prompt_durations = torch.tensor(note_seq.duration[:12]).unsqueeze(0).cuda().float()
# prompt_dts = torch.tensor(note_seq.dt[:12]).unsqueeze(0).cuda().float()
prompt_ids, prompt_velos, prompt_durations, prompt_dts = notedf2tensor(note_seq[:10], device="cuda")
pitch_gen, velo_gen, dur_gen, dt_gen = naive_greedy_decode(model,
                       prompt_ids, prompt_velos, prompt_durations, prompt_dts,
                                                           temperature=.5, max_gen_tokens=500)
tensor2midi(pitch_gen[0], velo_gen[0], dur_gen[0], dt_gen[0],
            savedir, "generated_midi_with_tempo_T0_5.mid")


#%%
note_seq_full = answers.sequences[0].cpu()
note_seq_full = note_seq_full[note_seq_full != PAD_ID]
for pitch in note_seq_full:
    print(librosa.midi_to_note(pitch), end=" ")




#%% save to midi and play
# create midi file based on the generated sequence
generated_midi = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
piano = pretty_midi.Instrument(program=piano_program)
delta = 0.25
for i, pitch in enumerate(note_seq_full):
    note = pretty_midi.Note(velocity=100, pitch=pitch.item(), start=i*delta, end=(i+1)*delta)
    piano.notes.append(note)
#%%
generated_midi.instruments.append(piano)
generated_midi.write(join(savedir, "generated_midi.mid"))


# %% Old version
# dataset = MusicScoreDataset(maestro_root, maestro_meta)
batch_size = 12
writer = SummaryWriter(savedir)
for epoch in trange(0, 100):
    rand_idx_seq = np.random.permutation(len(dataset))
    for i, csr in enumerate(trange(0, len(dataset), batch_size)):
        batch_idxs = rand_idx_seq[csr:csr + batch_size]
        batch_pitch, batch_velo, batch_dt, batch_duration = batch_sampler(dataset,
                                                                          batch_idxs, max_seq_len=512)
        out = model(batch_pitch.cuda(), labels=batch_pitch.cuda(), output_hidden_states=True)
        loss = out.loss
        last_hidden = out.hidden_states[-1]  # (batch, seq_len, embd)
        velocity_pred, duration_pred, dt_pred = readout(last_hidden)
        # note this is wrong, need to predict the next note so shift by 1
        loss_velo = F.mse_loss(velocity_pred, batch_velo.cuda())
        loss_duration = F.mse_loss(duration_pred, batch_duration.cuda())
        loss_dt = F.mse_loss(dt_pred, batch_dt.cuda())
        loss += loss_velo + loss_duration + loss_dt
        # compute additional loss based on last hidden state
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(
            f"epoch{epoch}-step{i:03d} {loss.item():.5f} vel {loss_velo.item():.5f} dur {loss_duration.item():.5f} dt {loss_dt.item():.5f}")
        writer.add_scalar("loss", loss.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("loss_velo", loss_velo.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("loss_duration", loss_duration.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("loss_dt", loss_dt.item(), epoch * len(dataset) // batch_size + i)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch * len(dataset) // batch_size + i)
        writer.add_scalar("epoch", epoch, epoch * len(dataset) // batch_size + i)
        # writer.add_scalar("grad_norm", torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),)

    torch.save(model.state_dict(), join(savedir, "ckpt", f"model_{epoch:03d}.pt"))

#%%





#%%# %%
# # load sample midi file
# midi_file = join(maestro_root, maestro_meta["midi_filename"][0].replace("/", os.path.sep))
# midi_data = pretty_midi.PrettyMIDI(midi_file)
#
# print(midi_data.instruments)
# len(midi_data.instruments[0].notes)
# # %%
# # create a new figure
# fig, ax = plt.subplots(figsize=(12, 4))
# # plot the piano roll
# plot_piano_roll(midi_data, 24, 84)
# plt.tight_layout()
# plt.show()
# # %%
# note_dt_dist = []
# note_len_dist = []
# note_velo_dist = []
# note_pitch_dist = []
# for i, note_sample in enumerate(midi_data.instruments[0].notes[1:]):
#     note_dt_dist.append(note_sample.start - midi_data.instruments[0].notes[i - 1].end)
#     note_len_dist.append(note_sample.duration)
#     note_velo_dist.append(note_sample.velocity)
#     note_pitch_dist.append(note_sample.pitch)
# # %%
#
# # %%
# plt.subplots(1, 3, figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.hist(note_len_dist, bins=5000)
# plt.xlim(0, 3)
# plt.title("Note Length Distribution")
# plt.subplot(1, 3, 2)
# plt.hist(note_velo_dist, bins=100)
# plt.title("Note Velocity Distribution")
# plt.subplot(1, 3, 3)
# plt.hist(note_pitch_dist, bins=100)
# plt.title("Note Pitch Distribution")
# plt.show()
#
# # %%
# print(len(maestro_meta.canonical_composer.unique()))
# print(len(maestro_meta.canonical_title.unique()))
#
# # %%
# notes_str = [librosa.midi_to_note(note.pitch)
#              for note in midi_data.instruments[0].notes]
# # %%
# midi_data.time_to_tick(midi_data.instruments[0].notes[-1].end)
# # %%