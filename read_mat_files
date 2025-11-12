# -*- coding: utf-8 -*-
# @author: hgy
# create time: 2025/11/12
import numpy as np
import scipy.io as scio
from scipy.signal import butter, filtfilt

class FileToData:
    def __init__(self, event, data):
        """get data from .mat files
        === parameters ===
        event: event file, .mat
        data: data file, .mat
        fs: sample rate, 500 to author's experiment.        
        """
        self.event = event
        self.data = data
        self.fs = 500

    def electrode_index(self, electrodes):
        """get electrodes index of EEG channels
        === parameters ===
        all_electrodes: electrodes of EEG cap.
        
        electrodes: the name of the EEG electrodes need. list of string.
        like:
            electrodes = ['Fp1', 'FC1']
        === return ===
        index: index of EEG electrodes input. list of int.
        
        """
        all_electrodes = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                          'FC5', 'FC1', 'FC2', 'FC6', 'PO3', 'T7', 'C3', 'Cz', 'C4', 'T8', 'PO4',
                          'CP5', 'CP1', 'CP2', 'CP6', 'PO7', 'P3', 'Pz', 'P4', 'PO8',
                          'POz', 'O1', 'Oz', 'O2']
        index = [i for i in range(len(all_electrodes)) if all_electrodes[i] in electrodes]

        return index

    def loss(self, t_label, p_label):
        """get the accuracy.
        === parameters ===
        t_label: target label
        p_label: predicted label
        
        === return ===
        loss: the accuracy of EEG
        
        """
        flag = 0
        if len(t_label) != len(p_label):
            print("y_label and p_label must have the same length")
            quit()
        for i in range(len(t_label)):
            if t_label[i] == p_label[i]:
                flag += 1

        return flag / len(t_label)


    def get_online_acc(self):
        """read online accuracy from .mat file.
        === return ===
        online_acc: the online accuracy of EEG.
        """
        acc = []
        for event in (self.event):
            # load total MAT data
            event_iter = scio.loadmat(event)['stimevent'][0][0]
            # get data we need
            right_label = [i.item() for i in event_iter['stimnum'][:, 0] if i != 0]  # delete zero item
            online_label = [i.item() for i in event_iter['stimnum'][:, 1] if i != 0]
            acc.append(np.round(100 * self.loss(right_label, online_label), 2))

        return acc, np.mean(acc)

    def data_mean(self, data1, data2):
        """average of two data.
        === return ===
        mean: the average of two data.
        """
        if np.shape(data1) != np.shape(data2):
            print("data must be same shape")
            quit()
        data = data1 + data2
        return np.array([i / 2 for i in data])

    # data concatenate
    def data_concatenate(self, data1, data2, keep_axis=0):
        return np.concatenate((data1, data2), axis=keep_axis)

    def EventData(self, event_i):
        '''
        获取采样时间点/采样数据点
        '''
        # load total MAT data
        event = scio.loadmat(event_i)['stimevent'][0][0]
        # get data we need
        toc = [i.item() for i in event['toc'][..., 0] if i != 0]  # delete zero item
        stimnum = [i.item() for i in event['stimnum'][:, 0] if i != 0]  # delete zero item
        return toc, stimnum

    def get_freq(self):
        '''
        获取刺激频率
        '''
        freq = scio.loadmat(self.event)['stimevent'][0][0]['fps'][0]
        return np.array(freq) # the default frequency matrix read is 2 dimensional.


    # load data matrix
    def EEGData(self, data_i):
        data_comp = scio.loadmat(data_i)['data_comp2']
        return data_comp


    def DataDeal(self, col_index, toc, data_comp, dots):
        data_use = []
        # column slice
        data_comp = np.array(data_comp[..., col_index])  # extract column

        for time_iter in toc:
            value = int(time_iter * self.fs)
            data = data_comp[int(value):int(value + dots), ...].T
            data_use.append(data)
        # print(f'shape of data_use: {np.shape(data_use)}')

        return data_use


    # prepare butetr bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, order=6):
        '''
        巴特沃斯带通滤波器
        lowcut: 低通滤波器截止频率
        highcut: 高通滤波器截止频率
        fs: 采样率
        '''
        if lowcut <= 0 or highcut <= 0:
            raise ValueError("lowcut and highcut must be positive")
        if lowcut >= highcut:
            raise ValueError("lowcut must be less than highcut")
        if self.fs <= 0:
            raise ValueError("Sampling frequency fs must be positive")
        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("data must be a list or numpy array")
        if np.array(data).size == 0:
            raise ValueError("data must not be empty")

        fa = 0.5 * self.fs
        low = lowcut / fa
        high = highcut / fa
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y


    # function of generate reference sin signal
    def generate_mscca_references(self, freqs, srate, T, phases, n_harmonics: int = 1):
        '''
        生成参考正余弦信号
        freqs: 频率
        srate: 采样率
        T: 信号长度
        phases: 相位
        n_harmonics: 正弦波的数量
        '''
        freqs = freqs.flatten()
        if isinstance(freqs, int) or isinstance(freqs, float):
            freqs = [freqs]
        freqs = np.array(freqs)[:, np.newaxis]
        if phases is None:
            phases = 0
        if isinstance(phases, int) or isinstance(phases, float):
            phases = [phases]
        phases = np.array(phases)[:, np.newaxis]
        t = np.linspace(0, T, int(T * srate))

        Yf = []
        for i in range(n_harmonics):
            if i % 2 == 0:
                Yf.append(np.stack([
                    np.sin(2 * np.pi * freqs * t + np.pi * phases),  # different phases pre-defined
                    np.cos(2 * np.pi * freqs * t + np.pi * phases),
                ], axis=1))
            else:
                Yf.append(np.stack([
                    np.sin(4 * np.pi * freqs * t + np.pi * phases),  # different phases pre-defined
                    np.cos(4 * np.pi * freqs * t + np.pi * phases),
                ], axis=1))

        Yf = np.concatenate(Yf, axis=1)
        return Yf


    def reference_s(self, freq, fs, time):
        '''
        创建参考正余弦信号
        '''
        return self.generate_mscca_references(freq, srate=fs, T=time, phases=None, n_harmonics=2)


# -----------------------------------load datas and labels-----------------------------------
    # data concatenate
    def filter_data(self, time, cap):
        sample_dot = time * self.fs
        low_pass = 3
        high_pass = 30

        filtered_temp = []
        label_temp = []
        # choose channels
        # ANT
        if cap == 'old':
            electrodes = ['PO7', 'P3', 'Pz', 'P4', 'POz', 'O1', 'Oz', 'O2']  # visual area electrodes
            # electrodes = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'Cz', 'CP5', 'CP6', 'CP1', 'CP2', 'CPz'] # motor area electrodes
            # electrodes = ['FC1', 'FC2', 'C3', 'C4', 'Cz', 'CP1', 'CP2', 'CPz'] # motor area electrodes
            cols = self.electrode_index(electrodes)
        # GREENTEK
        elif cap == 'new':
            electrodes = ['PO3', 'PO4', 'PO7', 'Pz', 'PO8', 'POz', 'O1', 'Oz', 'O2']  # visual area electrodes
            # electrodes = ['PO3', 'PO4', 'PO7', 'Pz', 'PO8', 'POz', 'O1', 'Oz'] # cross equipments for visual area electrodes
            # electrodes = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'Cz', 'CP5', 'CP6', 'CP1', 'CP2', 'CPz'] # motor area electrodes
            # electrodes = ['FC1', 'FC2', 'C3', 'C4', 'Cz', 'CP1', 'CP2', 'CPz'] # motor area electrodes
            cols = self.electrode_index(electrodes)

        for i, (event, data) in enumerate(zip(self.event, self.data)):
            toc, stimnum = self.EventData(event)
            data_tmp = self.EEGData(data)
            data_deal = self.DataDeal(cols, toc, data_tmp, sample_dot)
            if i == 0:
                filtered_temp = self.butter_bandpass_filter(data_deal, low_pass, high_pass, order=6)
                label_temp = stimnum
                # label_temp = [i-1 for i in label_temp]
            else:
                filtered_temp = self.data_concatenate(self.butter_bandpass_filter(data_deal, low_pass, high_pass, order=6),
                                                 filtered_temp, keep_axis=0)
                label_temp = self.data_concatenate(stimnum, label_temp, keep_axis=0)
                # label_temp = [i - 1 for i in label_temp]

        filtered_data = np.array(filtered_temp)
        label = np.array(label_temp)

        return filtered_data, label


    # data concatenate for raw data
    def raw_data(self, fs, time, cap):
        """read data without any preprocessing.
        """
        dot = time * fs
        raw_temp = []
        label_temp = []
        # choose channels
        # ANT
        if cap == 'old':
            electrodes = ['PO7', 'P3', 'Pz', 'P4', 'POz', 'O1', 'Oz', 'O2']
            cols = self.electrode_index(electrodes)
        # GREENTEK
        elif cap == 'new':
            electrodes = ['PO3', 'PO4', 'PO7', 'Pz', 'P4', 'POz', 'O1', 'Oz', 'O2']
            # electrodes = ['PO3', 'PO4', 'PO7', 'Pz', 'P4', 'POz', 'O1', 'Oz'] # cross equipments
            cols = self.electrode_index(electrodes)
        # all electrodes
        else:
            cols = [i for i in range(32)]

        for i, (e_i, d_i) in enumerate(zip(self.event, self.data)):
            toc, stimnum = self.EventData(e_i)
            data_tmp = self.EEGData(d_i)
            data_deal = self.DataDeal(cols, toc, data_tmp, dot, fs)
            if i == 0:
                raw_temp = data_deal
                label_temp = stimnum
            else:
                raw_temp = self.data_concatenate(data_deal, raw_temp, keep_axis=0)
                label_temp = self.data_concatenate(stimnum, label_temp, keep_axis=0)

        raw_data = np.array(raw_temp)
        label = np.array(label_temp)

        return raw_data, label


    def mean_data(self, data, label):
        """get average of data in the level of label.
        data: shape(n_trials, 1500, 8)
        label: shape(8, )
        return: shape(8, 1500)
               
        """
        
        # get unique label
        y_label = np.unique(label)
        data_1, data_2, data_3, data_4 = [], [], [], []

        # get data for each label
        for i in range(len(label)):
            if label[i] == y_label[0]:
                data_1.append(data[i])
            elif label[i] == y_label[1]:
                data_2.append(data[i])
            elif label[i] == y_label[2]:
                data_3.append(data[i])
            elif label[i] == y_label[3]:
                data_4.append(data[i])
        data_1 = np.stack([data[i] for i in range(len(label)) if label[i] == y_label[0]], axis=0)
        data_2 = np.stack([data[i] for i in range(len(label)) if label[i] == y_label[1]], axis=0)
        data_3 = np.stack([data[i] for i in range(len(label)) if label[i] == y_label[2]], axis=0)
        data_4 = np.stack([data[i] for i in range(len(label)) if label[i] == y_label[3]], axis=0)

        # shape(8,1500)
        data_1, data_2, data_3, data_4 = data_1.mean(axis=0), data_2.mean(axis=0), data_3.mean(axis=0), data_4.mean(axis=0)

        return data_1, data_2, data_3, data_4


    # data concatenate for feedback
    def feedback_data(self, fs, time, cap):
        dot = time * fs
        low_pass = 3
        high_pass = 30

        filtered_temp = []
        label_temp = []
        # ANT
        if cap == 'old':
            electrodes = ['PO7', 'P3', 'Pz', 'P4', 'POz', 'O1', 'Oz', 'O2']
            cols = self.electrode_index(electrodes)
        # GREENTEK
        elif cap == 'new':
            electrodes = ['PO3', 'PO4', 'PO7', 'Pz', 'P4', 'POz', 'O1', 'Oz', 'O2']
            # electrodes = ['PO3', 'PO4', 'PO7', 'Pz', 'P4', 'POz', 'O1', 'Oz'] # cross equipments
            cols = self.electrode_index(electrodes)

        for i, (event, data) in enumerate(zip(self.event, self.data)):
            toc, stimnum = self.EventData(event)
            data_tmp = self.EEGData(data)
            data_deal = self.DataDeal(cols, toc, data_tmp, dot, fs)
            if i == 0:
                filtered_temp = self.butter_bandpass_filter(data_deal, low_pass, high_pass, fs, order=6)
                label_temp = stimnum
            else:
                filtered_temp = self.data_concatenate(self.butter_bandpass_filter(data_deal, low_pass, high_pass, fs, order=6),
                                                 filtered_temp, keep_axis=0)
                label_temp = self.data_concatenate(stimnum, label_temp, keep_axis=0)

        feedback_data = np.array(filtered_temp)
        label = np.array(label_temp)

        return feedback_data, label
