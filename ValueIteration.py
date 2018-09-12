# @Author: soheil
# @Date:   2018-09-11 11:47:34
# @Last Modified by:   soheilred
# @Last Modified time: 2018-09-12 10:23:40
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys, os


def valueIteration(S, A, P, R, gamma):
	v2 = np.zeros([len(S)])
	v1 = np.zeros([len(S)])
	optimalAction = np.zeros([len(S)])
	i = 0
	eps = 0.01
	while (i is 0) or (np.linalg.norm(v2 - v1) > eps):
		i = i + 1
		v1 = np.copy(v2)
		for s in S:
			Q = np.zeros(len(A))
			for a in A:
				Q[a] = 0
				for sp in S:
					Q[a] = Q[a] + P[s, sp, a] * ( R[s, sp, a] + gamma*v1[sp] )
				print ('Q[a =',a ,',s =', s,'] =',"%.2f" % Q[a])
			v2[s] = np.max(Q)
			optimalAction[s] = np.argmax(Q)
		
		twodecimals = ["%.2f" % var for var in v2]
		print (twodecimals)
		print (i , ',', twodecimals)
		print (np.linalg.norm(v2 - v1) )
		print (optimalAction)
		return optimalAction


def main():
	# try:
	#     fileName = sys.argv[1]
	# except:
	#     fileName = input("Enter Filename: ")
	fileName = 'input1.csv'
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
	
	optimalAction = valueIteration(states, actions, P, R, gamma)


	df_out = pd.DataFrame(data={'idstate': np.array(list(states))\
		,'idaction': optimalAction}, columns=['idstate', 'idaction'])
	print (df_out.head())
	df_out.to_csv('out.csv')


if __name__ == "__main__": main()
# dist = numpy.linalg.norm(a-b)