maestro_v3_midi_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
maestro_v3_meta_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv"
maestro_v3_json_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json"
#%%
# download and unzip the urls above
# !wget $maestro_v3_midi_url
# !wget $maestro_v3_meta_url
# !wget $maestro_v3_json_url
# !unzip maestro-v3.0.0-midi.zip
#%%
import os
from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from torch.optim import AdamW, Adam
# from transformers import AdamW, get_linear_schedule_with_warmup
import pretty_midi
import librosa.display
import librosa
import matplotlib.pyplot as plt
maestro_root = r"E:\Datasets\maestro-v3.0.0-midi\maestro-v3.0.0"
# visualize the midi file as piano roll
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                 hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                 fmin=pretty_midi.note_number_to_hz(start_pitch))

# load the metadata
maestro_meta = pd.read_csv(join(maestro_root, "maestro-v3.0.0.csv"))
#%%
# load sample midi file
midi_file = join(maestro_root, maestro_meta["midi_filename"][0].replace("/", os.path.sep))
midi_data = pretty_midi.PrettyMIDI(midi_file)

print(midi_data.instruments)
len(midi_data.instruments[0].notes)
#%%
# create a new figure
fig, ax = plt.subplots(figsize=(12, 4))
# plot the piano roll
plot_piano_roll(midi_data, 24, 84)
plt.tight_layout()
plt.show()
#%%
note_dt_dist = []
note_len_dist = []
note_velo_dist = []
note_pitch_dist = []
for i, note_sample in enumerate(midi_data.instruments[0].notes[1:]):
    note_dt_dist.append(note_sample.start - midi_data.instruments[0].notes[i-1].end)
    note_len_dist.append(note_sample.duration)
    note_velo_dist.append(note_sample.velocity)
    note_pitch_dist.append(note_sample.pitch)
#%%

#%%
plt.subplots(1, 3, figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(note_len_dist, bins=5000)
plt.xlim(0, 3)
plt.title("Note Length Distribution")
plt.subplot(1, 3, 2)
plt.hist(note_velo_dist, bins=100)
plt.title("Note Velocity Distribution")
plt.subplot(1, 3, 3)
plt.hist(note_pitch_dist, bins=100)
plt.title("Note Pitch Distribution")
plt.show()

#%%
print(len(maestro_meta.canonical_composer.unique()))
print(len(maestro_meta.canonical_title.unique()))

#%%
notes_str = [librosa.midi_to_note(note.pitch)
             for note in midi_data.instruments[0].notes]
#%%
midi_data.time_to_tick(midi_data.instruments[0].notes[-1].end)
#%%
# https://notebook.community/craffel/pretty-midi/Tutorial
# Each note has a start time, end time, pitch, and velocity.
#
# Velocity / volume of a note: 1-127
# Pitch of a note: 0-127 C1-G9
# Duration of a note: seconds, float number
# Start time of a note: seconds, float number

#%%

#%% Scratch
# midi to mp3
# !pip install mido
# !pip install midi2audio
#%%
import fluidsynth
# from pydub import AudioSegment

# Load the MIDI file using FluidSynth
fs = fluidsynth.Synth()
fs.start(driver="coreaudio") # Use appropriate driver for your system
sfid = fs.sfload(midi_file)
fs.program_select(0, sfid, 0, 0)
# Render the MIDI file to a WAV file
fs.midi_to_audio("example.wav", "example.mid")