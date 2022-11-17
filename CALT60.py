# -*- coding: utf-8 -*-
"""
@file      :  CALT60.py
@Time      :  2022/11/17 17:07
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""
import glob

import numpy as np
from acoustics.room import t60_impulse
from scipy.io import wavfile
from scipy import stats


def cal_t60(file_name):
    """
    Reverberation time from a WAV impulse response.
    :param file_name: name of the WAV file containing the impulse response.
    :returns: Reverberation time :math:`T_{60}`

    """
    fs, raw_signal = wavfile.read(file_name)

    # init = -5.0
    # end = -35.0
    # factor = 2.0
    init = -5.0
    end = -25.0
    factor = 3.0

    abs_signal = np.abs(raw_signal) / np.max(np.abs(raw_signal))

    # Schroeder integration
    sch = np.cumsum(abs_signal[::-1] ** 2)[::-1]
    sch = np.clip(sch, 1e-8, 1e+8)
    xxx = np.clip(np.max(sch), 1e-8, 1e+8)
    sch_db = 10.0 * np.log10(sch / xxx)

    # Linear regression
    sch_init = sch_db[np.abs(sch_db - init).argmin()]
    sch_end = sch_db[np.abs(sch_db - end).argmin()]
    init_sample = np.where(sch_db == sch_init)[0][0]
    end_sample = np.where(sch_db == sch_end)[0][0]
    x = np.arange(init_sample, end_sample + 1) / fs
    y = sch_db[init_sample:end_sample + 1]
    slope, intercept = stats.linregress(x, y)[0:2]

    # Reverberation time (T30)
    db_regress_init = (init - intercept) / slope
    db_regress_end = (end - intercept) / slope
    t60 = factor * (db_regress_end - db_regress_init)

    return t60


if __name__ == "__main__":
    # 分频段算t60的方法 -- 不太行
    # res = t60_impulse(file_name="/Users/bajianxiang/Desktop/internship/放一些小程序/RirData/6BatteryQuarles_right.wav",
    #             bands = np.array([125, 250, 500, 1000, 2000, 4000]),
    #             rt='t30')
    # print(res)
    for file in glob.glob('/Users/bajianxiang/Desktop/internship/放一些小程序/RirData/*.wav'):
        t60 = cal_t60(file)
        freq_t60 = t60_impulse(file, np.array([125, 250, 500, 1000, 2000, 4000]))
        print('file:', file.split('.wav')[0].split('/')[-1], '\nfreq_t60:', freq_t60, 't60:', t60)
