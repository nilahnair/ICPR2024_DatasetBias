# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:22:12 2024

@author: nilah
"""
import os

base_dir = '/results/'

def create_folder():
    for i in range(1,38):
        folder = 'exp' + str(i)
        path = os.path.join(base_dir, folder)
        try:
            os.mkdir(path)
        except OSError as error:  
            print(error)
        path2 = os.path.join(path, 'cnn')
        path3 = os.path.join(path2, 'plots')
        try:
            os.mkdir(path2)
            os.mkdir(path3)
        except OSError as error:  
            print(error)
        path4 = os.path.join(path, 'lstm')
        path5 = os.path.join(path4, 'plots')
        try:
            os.mkdir(path4)
            os.mkdir(path5)
        except OSError as error:  
            print(error)
        path6 = os.path.join(path, 'cnntrans')
        path7 = os.path.join(path6, 'plots')
        try:
            os.mkdir(path6)
            os.mkdir(path7)
        except OSError as error:  
            print(error)
    return 
        
        
if __name__ == '__main__':
    create_folder()
    print('done')