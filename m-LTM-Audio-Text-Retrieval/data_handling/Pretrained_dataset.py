#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
import torch
import random
import librosa
from torch.utils.data import Dataset, DataLoader, DistributedSampler, BatchSampler
import torch.nn.functional as F
import os
import torchaudio


from re import sub


def text_preprocess(sentence):

    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')
    return sentence
# dataset_folder = "/home/tienluong/multi-modal/dataset/AudioSet_waveform"

def _load_json_file(files, dataset_folder, blacklist=None):
    json_data = []
    audio_id = 0
    if blacklist is not None:
        with open(blacklist, 'r') as f:
            blacklist = json.load(f)
    for file in files:
        with open(file, "r") as f:
            json_obj = json.load(f)
            if json_obj["num_captions_per_audio"] == 1:
                for item in json_obj["data"]:
                    if "FreeSound" in file and blacklist is not None:
                        if item["id"] in blacklist["FreeSound"]:
                            continue
                    elif ("AudioSet" in file or "AudioCaps" in file) and blacklist is not None:
                        if item["id"] in blacklist["AudioSet"]:
                            continue
                    # temp_dict = {"audio": item["audio"], "caption": item["caption"], "id": audio_id,
                    #              "duration": item["duration"]}
                    temp_dict = {"audio": os.path.join(dataset_folder, item["id"]), "caption": item["caption"], "id": audio_id,
                                 "duration": item["duration"]}
                    json_data.append(temp_dict)
                    audio_id += 1
            else:
                for item in json_obj["data"]:
                    if "Clotho" in file and blacklist is not None:
                        if item["id"] in blacklist["FreeSound"]:
                            continue
                    for i in range(1, json_obj["num_captions_per_audio"] + 1):
                        temp_dict = {"audio": item["audio"], "caption": item[f"caption_{i}"], "id": audio_id,
                                     "duration": item["duration"]}
                        json_data.append(temp_dict)
                    audio_id += 1
    return json_data


class AudioLanguagePretrainDataset(Dataset):

    def __init__(self, json_files, audio_config, dataset_folder,blacklist=None):

        self.json_data = _load_json_file(json_files, dataset_folder,blacklist)
        self.lengths = [item["duration"] for item in self.json_data]

        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = audio_config["max_length"] * self.sr
        else:
            self.max_length = 0

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):

        item = self.json_data[index]
        wav_path = item["audio"]
        waveform, _ = librosa.load(wav_path, sr=self.sr, mono=True)
        audio_name = os.path.basename(wav_path)

        if self.max_length != 0:
            # if audio length is longer than max_length, we randomly crop it to mac length
            if waveform.shape[-1] > self.max_length:
                max_start = waveform.shape[-1] - self.max_length
                start = random.randint(0, max_start)
                waveform = waveform[start: start + self.max_length]

        caption = text_preprocess(item["caption"])
        audio_id = item["id"]

        return torch.tensor(waveform), caption, audio_id, audio_name
        # return duration, caption, audio_id


def collate_fn(batch):
    wav_list = []
    text_list = []
    audio_idx_list = []
    audio_name = []
    max_length = max([i[0].shape[-1] for i in batch])
    # for waveform, text, audio_idx in batch:
    for waveform, text, audio_idx, n in batch:
        if waveform.shape[-1] < max_length:
            pad_length = max_length - waveform.shape[-1]
            waveform = F.pad(waveform, [0, pad_length], "constant", 0.0)
        wav_list.append(waveform)
        text_list.append(text)
        audio_idx_list.append(audio_idx)
        audio_name.append(n)

    waveforms = torch.stack(wav_list, dim=0)
    audio_idx = torch.tensor(audio_idx_list).type(torch.long)
    return waveforms, text_list, audio_idx, audio_name


def pretrain_dataloader(config):
    dataset = AudioLanguagePretrainDataset(config["json_files"], config["wav"], config["dataset_folder"], config["blacklist"])

    return DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

if __name__ =="__main__":
    from tools.config_loader import get_config
    config_f = "rebuttal-wavcaps"
    dataset_folder = "/home/tienluong/multi-modal/dataset/AudioSet_waveform"
    config = get_config(config_f)

    dataloader = pretrain_dataloader(config)
    wave, text, _, _ = next(iter(dataloader))
    print(wave[0])
    print(text[0])
