# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def ler(fileName):
    # dataset = pd.read_csv(fileName,delimiter =';')
    dataset = np.array(pd.read_csv(fileName, header=None)).T[0]
    return dataset