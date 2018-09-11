# @Author: soheil
# @Date:   2018-09-11 11:47:34
# @Last Modified by:   soheil
# @Last Modified time: 2018-09-11 13:53:44
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys, os


def valueIteration(S, A, P, R, gamma):
    pass

def main():
    # try:
    #     fileName = sys.argv[1]
    # except:
    #     fileName = input("Enter Filename: ")
    fileName = 'input.csv'
    df = pd.read_csv(fileName)
    print (df.head())
    df.to_csv('out.csv')


if __name__ == "__main__": main()
# PROJECT_PATH = os.path.abspath(os.path.split(sys.argv[0])[0])