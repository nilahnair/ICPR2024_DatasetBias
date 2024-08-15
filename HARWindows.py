'''
Created on May 18, 2019

@author: fmoya
modified and reused: nilah 
'''

import os
import numpy as np
from tqdm import tqdm
from random import choices

from torch.utils.data import Dataset

import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset

import pandas as pd
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class HARWindows(Dataset):
    '''
    classdocs
    '''


    def __init__(self, config, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = config
        self.harwindows = pd.read_csv(csv_file)
        self.root_dir = root_dir
        #self.transform = transform

    def __len__(self):
        return len(self.harwindows)

    def __getitem__(self, idx):
        '''
        get single item

        @param data: index of item in List
        @return window_data: dict with sequence window, label of window, and labels of each sample in window
        '''
        window_name = os.path.join(self.root_dir, self.harwindows.iloc[idx, 0])

        f = open(window_name, 'rb')
        data = pickle.load(f, encoding='bytes')
        f.close()

        '''cross-verify the names with the preprocessing files used.
        in the function generate_data, 'Creating sequence file number {} with id {}'.format(f, counter_seq))
        # Storing the sequences
        obj = {"data": seq, "label": y[f], "labels": y_all[f]}
        made modification here if required.'''

        X = data['data']
        y = data['label']
        Y = data['labels']
        
        window_data = {"data": X, "label": y, "labels": Y}
        return window_data
        
    def _random_curve(self, window_len: int, sigma=0.05, knot=4):
        """
        Generates a random cubic spline with mean value 1.0.
        This curve can be used for smooth, random distortions of the data, e.g., used for time warping.

        Note: According to T. Um, a cubic splice is not the best approach to generate random curves.
        Other aprroaches, e.g., Gaussian process regression, Bezier curve, etc. are also reasonable.

        :param window_len: Length of the data window (for example, 100 frames), the curve will have this length
        :param sigma: sigma of the curve, the spline deviates from a mean of 1.0 by +- sigma
        :param knot: Number of anchor points
        :return: A 1d cubic spline
        """

        random_generator = np.random.default_rng()

        xx = (np.arange(0, window_len, (window_len - 1) / (knot + 1))).transpose()
        yy = random_generator.normal(loc=1.0, scale=sigma, size=(knot + 2, 1))
        x_range = np.arange(window_len)
        cs_x = CubicSpline(xx, yy)
        return cs_x(x_range).flatten()



   