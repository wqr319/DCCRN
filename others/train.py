import argparse
import os

import librosa
import numpy as np
import torch
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from conv_stft import ConvSTFT
from dc_crn import DCCRN


def load_data_list(folder='./dataset', setname='train'):
    assert (setname in ['train', 'val'])
    dataset = {'innames': [], 'outnames': [], 'shortnames': []}
    foldername = folder + '/' + setname + 'set'
    print("Loading files...")
    filelist = os.listdir("%s_noisy" % foldername)
    filelist = [f for f in filelist if f.endswith('.wav')]
    for i in tqdm(filelist):
        dataset['innames'].append("%s_noisy/%s" % (foldername, i))  # noisy wav 路径
        dataset['outnames'].append("%s_clean/%s" % (foldername, i))  # clean wav 路径
        dataset['shortnames'].append("%s" % i)  # 此数据集noisy/clean wav文件名相同，此处为文件名
    return dataset


def load_data(dataset):
    print("Loading audiodata...")
    dataset['inaudio'] = [None] * len(dataset['innames'])
    dataset['outaudio'] = [None] * len(dataset['outnames'])
    for id in tqdm(range(len(dataset['innames']))):  # 此数据集noisy/clean wav文件名相同
        if (dataset['inaudio'][id] is None) and (dataset['outaudio'][id] is None):
            inputdata, sr = librosa.load(dataset['innames'][id], sr=None)
            inputdata = librosa.resample(inputdata, sr, 16000)
            outputdata, sr = librosa.load(dataset['outnames'][id], sr=None)
            outputdata = librosa.resample(outputdata, sr, 16000)
            dataset['inaudio'][id] = np.float32(inputdata)
            dataset['outaudio'][id] = np.float32(outputdata)
    return dataset


class AudioDataset(data.Dataset):
    """
    Audio sample reader
    """

    def __init__(self, data_type):
        dataset = load_data_list(setname=data_type)
        self.dataset = load_data(dataset)
        self.file_names = dataset['shortnames']

    def __getitem__(self, index):
        mixed = torch.from_numpy(self.dataset['inaudio'][index]).type(torch.FloatTensor)
        clean = torch.from_numpy(self.dataset['outaudio'][index]).type(torch.FloatTensor)
        return mixed, clean

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat(self, inputs):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :inp.shape[0]] = inp
        return input_mat

    def collate_fn(self, inputs):
        noisys, cleans = zip(*inputs)
        print(noisys)
        seq_lens = torch.IntTensor([i.shape[0] for i in noisys])
        noisys = torch.FloatTensor(self.zero_pad_concat(noisys))
        cleans = torch.FloatTensor(self.zero_pad_concat(cleans))
        batch = [noisys, cleans, seq_lens]
        return batch


def train():
    loss_epochs = []
    for epoch in range(args.num_epochs):
        loss_epoch = 0
        num_steps = 0
        train_bar = tqdm(train_data_loader)
        for input in train_bar:
            torch.cuda.empty_cache()  # 释放显存
            noisy, clean, seq_len = map(lambda x: x.cuda(), input)
            out_spec, out_wav = net(noisy)
            for i, l in enumerate(seq_len):
                out_wav[i, l:] = 0
            clean_spec = stft(clean)
            # 每批语音si-snr的平均值负数
            loss = net.loss(out_wav, clean[:, :out_wav.shape[1]], loss_mode='SI-SNR')
            optimizer.zero_grad()  # 清除梯度
            loss.backward()
            loss_epoch += loss.item()
            num_steps += 1
            optimizer.step()  # 训练
        loss_epochs.append(loss_epoch / num_steps)
        print("Average loss per wav of %s epoch: %s" % (epoch + 1, loss_epoch / num_steps))
        # print("Steps per epoch:%s" % num_steps)
    torch.save(net.state_dict(), (args.model_dir + 'new_params_epoch_%s.pt') % args.num_epochs)
    print(loss_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, type=int, help='train batch size')
    parser.add_argument('--num_epochs', default=10, type=int, help='train epochs number')
    parser.add_argument('--model_dir', default='./params/', help='Directory containing params.json?')
    parser.add_argument('--win_len', default=400, type=int, help='frame length')
    parser.add_argument('--win_inc', default=100, type=int, help='hop length')
    parser.add_argument('--fft_len', default=512, type=int, help='FFT length')
    args = parser.parse_args()

    stft = ConvSTFT(args.win_len, args.win_inc, args.fft_len, 'hann', 'complex').cuda()

    train_dataset = AudioDataset(data_type='val')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                   collate_fn=train_dataset.collate_fn, shuffle=True, num_workers=0)

    net = DCCRN(rnn_units=256, masking_mode='E', use_clstm=False, kernel_num=[32, 64, 128, 256, 256, 256]).cuda()  # 待调整
    optimizer = optim.Adam(net.parameters(), lr=1e-3)  # 待调整
    train()
