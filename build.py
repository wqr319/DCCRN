import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
import os
import tqdm
import pysepm
import librosa
from pytorch_lightning import callbacks
import torch.nn.functional as F

from dc_crn import DCCRN

class Plmodule(pl.LightningModule):
    def __init__(self):
        super(Plmodule, self).__init__()
        self.model = DCCRN()

    def training_step(self,batch):
        noisy, clean = batch
        out_spec, out_wav = self.model(noisy)
        loss = self.model.loss(out_wav, clean)
        self.log('train_loss',loss,on_step=False,on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        csig_list, cbak_list, covl_list = [], [], []
        pesq_list = []
        name_list = os.listdir('/home/wqr/vbd/noisy_testset_wav')
        for name in tqdm.tqdm(name_list):
            noisy_audio = librosa.load(os.path.join('/home/wqr/vbd/noisy_testset_wav', name),
                                       sr=16000)[0]
            target_audio = librosa.load(os.path.join('/home/wqr/vbd/clean_testset_wav', name),
                                        sr=16000)[0]
            noisy_audio = torch.from_numpy(noisy_audio).cuda().unsqueeze(0)
            _, wav_out = self.model(noisy_audio)
            wav_out = wav_out.cpu().squeeze().detach().numpy()
            target_audio = target_audio[:len(wav_out)]
            csig, cbak, covl = pysepm.composite(target_audio,wav_out,16000)
            pesq = pysepm.pesq(target_audio,wav_out,16000)[1]
            csig_list.append(csig)
            cbak_list.append(cbak)
            covl_list.append(covl)
            pesq_list.append(pesq)
            def mean(lst):
                return sum(lst)/len(lst)
        self.log('csig',mean(csig_list))
        self.log('cbak',mean(cbak_list))
        self.log('covl',mean(covl_list))
        self.log('pesq',mean(pesq_list))
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=1e-3)
        return optimizer


def build_callbacks():
    ###################### callbacks #########################################
    my_callbacks = [
        callbacks.ModelCheckpoint(
            monitor='csig',
            mode='max',
            dirpath='log/',
            filename='{epoch}-{csig:.3f}'
        )
    ]
    logger = loggers.TensorBoardLogger(
        save_dir='log',
    )
    return my_callbacks, logger


def build_trainer():
    my_callbacks, logger = build_callbacks()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        logger=logger,
        callbacks=my_callbacks,
        fast_dev_run=False,
        log_every_n_steps=50,
        max_epochs=300,
        gradient_clip_val=1,
        strategy='ddp' if torch.cuda.device_count()>1 else None,
    )
    return trainer