import os

import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import h5py
import librosa

class AudioDataset(Dataset):
    def __init__(self):
        self.noisylist = os.listdir('/home/wqr/vbd/clean_trainset_28spk_wav/')
        self.cleanlist = os.listdir('/home/wqr/vbd/noisy_trainset_28spk_wav/')
        self.noisylist.sort()
        self.cleanlist.sort()
        assert len(self.noisylist)==len(self.cleanlist)

    def read_clips(self,fileid):
        noisy_audio = librosa.load(os.path.join('/home/wqr/vbd/clean_trainset_28spk_wav',
                                                self.noisylist[fileid]),sr=16000)[0]
        target_audio = librosa.load(os.path.join('/home/wqr/vbd/noisy_trainset_28spk_wav',
                                                 self.cleanlist[fileid]),sr=16000)[0]
        return noisy_audio, target_audio

    def __getitem__(self, item):
        noisy_audio, target_audio = self.read_clips(fileid=item)
        noisy_audio = torch.from_numpy(noisy_audio)
        target_audio = torch.from_numpy(target_audio)
        if len(noisy_audio) < 16000*3:
            noisy_audio = F.pad(noisy_audio,(0,16000*3-len(noisy_audio)))
            target_audio = F.pad(target_audio,(0,16000*3-len(target_audio)))
        else:
            noisy_audio = noisy_audio[:16000*3]
            target_audio = target_audio[:16000*3]
        return noisy_audio, target_audio

    def __len__(self):
        return len(self.noisylist)

class H5Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.hdf5_dir = '../data_source/data.hdf5'
        self.f = h5py.File(self.hdf5_dir, 'r')
        self.noisy_ds = self.f['noisy_ds']
        self.target_ds = self.f['target_ds']
    def __getitem__(self, item):
        noisy_audio = self.noisy_ds[item]
        target_audio = self.target_ds[item]
        return noisy_audio,target_audio
    def __len__(self):
        return (10 * 3600 // 3)

def get_loader():
    train_loader = DataLoader(H5Dataset(),batch_size=8,shuffle=True,num_workers=4)
    return train_loader