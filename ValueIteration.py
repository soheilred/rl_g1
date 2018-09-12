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
    gamma = 0.9
    actions = set()
    states = set()
    for i in range(df.shape[0]):
    	states.add(df.at[i,'idstatefrom'])
    	states.add(df.at[i,'idstateto'])
    	actions.add(df.at[i,'idaction'])

    P = np.empty([len(states), len(states), len(actions)])
    R = np.empty([len(states), len(states), len(actions)])
    # print(P.shape)

    for i in range(df.shape[0]):
    	P[df.at[i, 'idstatefrom'], df.at[i,'idstateto'], df.at[i,'idaction']] = df.at[i,'probability']
    	R[df.at[i, 'idstatefrom'], df.at[i,'idstateto'], df.at[i,'idaction']] = df.at[i,'reward']
    
    v = np.zeros([len(states)])
    for s in states:
		Q = np.zeros(len(actions))
		for a in actions:
			Q[a] = 0
			for sp in states:
				Q[a] = Q[a] + P[s, sp, a] * ( R[s, sp, a] + gamma*v[sp] )
		v[s] = np.max(Q)
    
    print (v)

    # print (df.head())
    df.to_csv('out.csv')


if __name__ == "__main__": main()
# dist = numpy.linalg.norm(a-b)