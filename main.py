import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import fftpack
from scipy.io import loadmat, savemat
from scipy.stats import kurtosis
from torch import optim, sign, log, norm, nn
from torch.utils.data import TensorDataset, DataLoader
from Functions import *

from Model.Net import Net
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use_gpu = torch.cuda.is_available()
use_gpu = False


def random_seed(num: int):
    # region 随机种子设置
    seed = int(num)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.


def funcEnv_Es(signal_pp, Title1, Fs, SignalLim, EsLim):
    a_signal_raw = fftpack.hilbert(signal_pp)
    hx_raw_abs = abs(a_signal_raw)
    hx_raw_abs1 = hx_raw_abs - np.mean(hx_raw_abs)
    es_raw = fftpack.fft(hx_raw_abs1, len(signal_pp))
    es_raw_abs = abs(es_raw) * 2 / len(hx_raw_abs)

    t1 = (np.arange(0, len(signal_pp))) / Fs

    f1 = t1 * Fs * Fs / len(signal_pp)

    plt.figure()
    plt.subplot(211)
    plt.plot(t1, signal_pp, label=u"raw signal")
    plt.xlim(SignalLim)
    plt.title(Title1)
    plt.subplot(212)
    plt.plot(f1, es_raw_abs)
    plt.xlabel('Envelope spectrum')
    plt.xlim(EsLim)
    plt.show()
    return es_raw_abs




def train_CNN(dataloader, epochs , n_filter, lambd, lr, fs):

    model = Net(n_filter, fs)
    if use_gpu:
        model = model.cuda()

    optimer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimer, 50, eta_min=1e-6)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimer, step_size=50, gamma=0.1)

    # Kurt
    lossFn = funcKurtosis


    loss_list = []
    min_loss = 10000

    for epoch in range(epochs):
        totalloss = 0
        for idx, batch_x in enumerate(dataloader):
            if use_gpu:
                batch_x = batch_x[0].cuda()
            else:
                batch_x = batch_x[0]
            y_predict, es_predict = model(batch_x)

            lplq_loss = g_lplq(es_predict, 2, 3)

            ## Multi-task
            loss = -lossFn(y_predict, n_filter // 2) + lambd * lplq_loss




            totalloss += loss.item()
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            lr_scheduler.step()


        if totalloss == 0.0:
            break
        if not os.path.exists('model_save'):
            os.mkdir('model_save')

        if totalloss < min_loss:
            min_loss = totalloss
            torch.save(model.state_dict(), 'model_save/checkpoint_sig2_kurt.pth')
        loss_list.append(totalloss)

    # plt.figure()
    # plt.plot(np.linspace(0, epochs, epochs), loss_list)
    # plt.show()

    # plt.figure()
    # plt.plot(np.linspace(0, epochs, epochs), alpha_es)
    # plt.show()

def plot_CNN(x, n_filter):
    # f = DENet()
    myNet = Net(n_filter)
    best_model_dict = torch.load('model_save/checkpoint_sig2_kurt.pth', map_location=torch.device('cpu'))
    myNet.load_state_dict(best_model_dict)
    myNet.eval()
    x = torch.tensor(x, dtype=torch.float)
    y, _ = myNet(x)
    rec = y.detach().numpy().squeeze().reshape(-1)
    es = funcEnv_Es(rec, 'Filterd Signal', 20000, [0, 1], [0, 1000])
    return rec, es

if __name__ == '__main__':
    random_seed(42)
    ## simulate
    sig = loadmat('data/sig2.mat')
    x = sig['x']
    x = x - np.mean(x)
    BPFI = sig['BPFI']
    fs = 20000
    learning_rate = 0.01
    epochs = 200
    n_filter = 80
    lambd = 0.01
    ## cwru ball
    # sig = loadmat('data/2HP/12k_Drive_End_OR021@6_2_236.mat')
    # x = sig['X236_DE_time']
    # x = x - np.mean(x)
    # x = x[24000:48000, 0]
    # fs = 12000
    # learning_rate = 0.01
    # epochs = 200
    # n_filter = 80
    # lambd = 0.01

    ## XJTU
    # sig = loadmat('data/Bearing2_1.mat')
    # dat = sig['x']
    # learning_rate = 1e-2
    # epochs = 200
    # n_filter = 80
    # lambd = 0.01
    # fs = 25600
    # signal = np.zeros((dat.shape[0], dat.shape[1]))
    # es = np.zeros((dat.shape[0], dat.shape[1]))
    # for idx, x in enumerate(dat):
    #     x = x.reshape(1, 1, -1)
    #     # dat = np.array(x).squeeze()
    #     # dat = dat[:, np.newaxis, :]
    #     dataset = TensorDataset(torch.tensor(x, dtype=torch.float))
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    #
    #     train_CNN(dataloader, epochs, n_filter, lambd, learning_rate, fs)
    #     rec, e = plot_CNN(x, n_filter)
    #     signal[idx, :] = rec
    #     es[idx, :] = e
    #     print('Complete %d' % (idx + 1))
    #
    # rec_dict = {"x": signal,
    #             "es": es}
    # savemat('results/bearing2_1_cnnbd_new.mat', rec_dict)

    # one signal
    x = x.reshape(1, 1, -1)
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    start = time.time()
    train_CNN(dataloader, epochs, n_filter, lambd, learning_rate, fs)
    run = time.time() - start
    print(run)
    rec, e = plot_CNN(x, n_filter)

    rec_dict = {"x": rec,
                "es": e}
    if not os.path.exists('results'):
        os.mkdir('results')
    savemat('results/sig2_mnnbd.mat', rec_dict)
