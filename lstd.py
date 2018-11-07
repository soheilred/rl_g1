# @Author: soheil
# @Date:   2018-09-11 11:47:34
# @Last Modified by:   soheilred
# @Last Modified time: 2018-10-14 22:13:30
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys, os
import time

class MDP(object):
	"""docstring for MDP"""
	def __init__(self):
		self.actions = set()
		self.states = set()
		self.P = None
		self.R = None
		self.v = None
		self.v0 = None
		self.gamma = 0.9
		self.samples = []
		self.n_episode = 50

	def sampleGenerator(self, state, samples):
		# np.random.seed(89)

		n = 4
		s = range(n)
		a = range(4) # 0 for right, 1 for up, 2 for left, 3 for down

		if (state == s[0]):
			if (np.random.rand() < .5):
				action = a[0]
				reward = -10
				sp = s[1]
				sample = [state, action, sp, reward]
				samples.append(sample)
				self.sampleGenerator(sp, samples)
			else:
				action = a[1]
				reward = 0
				sp = s[2]
				sample = [state, action, sp, reward]
				samples.append(sample)
				self.sampleGenerator(sp, samples)

		elif (state == s[1]):
			if (np.random.rand() < .5):
				action = a[1]
				reward = 10
				sp = s[3]
				sample = [state, action, sp, reward]
				samples.append(sample)
				self.sampleGenerator(sp, samples)
			else:
				action = a[2]
				reward = 0
				sp = s[0]
				sample = [state, action, sp, reward]
				samples.append(sample)
				self.sampleGenerator(sp, samples)

		elif (state == s[2]):
			if (np.random.rand() < .5):
				action = a[0]
				reward = 10
				sp = s[3]
				sample = [state, action, sp, reward]
				samples.append(sample)
				self.sampleGenerator(sp, samples)
			else:
				action = a[3]
				reward = 0
				sp = s[0]
				sample = [state, action, sp, reward]
				samples.append(sample)
				self.sampleGenerator(sp, samples)

		elif (state == s[3]):
			pass

	def makeSamples(self, n):
		state = 0
		self.samples = []
		self.sampleGenerator(state, self.samples)
		self.samplesToCSV(n)

		
	def samplesToCSV(self, n):
		import csv
		# print (self.samples)
		sample_out = list(self.samples)
		sample_out.insert(0,["idstatefrom", "idaction", "idstateto","reward"])
		sampleFile = "./samples/sampleSet_" + str(n) +".csv"
		with open(sampleFile, "w") as f:
			writer = csv.writer(f)
			writer.writerows(sample_out)

	# def csvToSamples(self):
	# 	import csv

			
	def lstd(self):
		m = 2 # number of features
		A = np.zeros([m,m])
		b = np.zeros([m,1])
		# phi_k0 = np.zeros([m,1])
		# phi_k1 = np.zeros([m,1])
		for i in range(self.n_episode):
			self.makeSamples(i)
			for sample in self.samples:
				phi_k0 = self.phi(sample[0])
				phi_k1 = self.phi(sample[2])
				A += np.dot(phi_k0, np.transpose(phi_k0 - self.gamma*phi_k1))
				b += phi_k0*sample[3]
				# print ("phi_k0 " , phi_k0.shape , "b shape " , b.shape)
				# print ("A " , A , "\nb " , b)
			t = float(len(self.samples))
			try:
				A_inverse = np.linalg.inv(A)
				# print ("A^-1 = " , A_inverse)
				print ("theta_t = ", np.transpose(np.dot(t*A_inverse, (1.0/t)*b)))
			except np.linalg.LinAlgError:
				# Not invertible. Skip this one.
				print ("ERROR, A is not invertible")
				pass


			
	def phi(self, state):
		if (state == 0):
			return np.array([1.0,1.0]).reshape(2,1)
		elif (state == 1):
			return np.array([2.0,1.0]).reshape(2,1)
		elif (state == 2):
			return np.array([1.0,2.0]).reshape(2,1)
		elif (state == 3):
			return np.array([2.0,2.0]).reshape(2,1)


	def readInput(self):
		fileName = 'inventory_100_100_0.csv'
		df = pd.read_csv(fileName)
		for i in range(df.shape[0]):
			self.states.add(df.at[i,'idstatefrom'])
			self.states.add(df.at[i,'idstateto'])
			self.actions.add(df.at[i,'idaction'])

		self.P = np.empty([len(self.states), len(self.states), len(self.actions)])
		self.R = np.empty([len(self.states), len(self.states), len(self.actions)])

		for i in range(df.shape[0]):
			self.P[df.at[i, 'idstatefrom'], df.at[i,'idstateto'], df.at[i,'idaction']] = df.at[i,'probability']
			self.R[df.at[i, 'idstatefrom'], df.at[i,'idstateto'], df.at[i,'idaction']] = df.at[i,'reward']
		
		print ("states shape: " , len(self.states))
		print ("actions shape: " ,len(self.actions))
		print ("P shape: " , self.P.shape)
		print ("R shape: " , self.R.shape)



	def valueIteration(self):
		self.readInput()
		start_time = time.time()
		v2 = np.zeros([len(self.states)])
		v1 = np.zeros([len(self.states)])
		optimalAction = np.zeros([len(self.states)])
		i = 0
		eps = 0.01
		while (i is 0) or (np.linalg.norm(v2 - v1) > eps):
			i =  1
			v1 = np.copy(v2)
			for s in self.states:
				Q = np.zeros(len(self.actions))
				for a in self.actions:
					Q[a] = sum(self.P[s,:, a] * ( self.R[s,:, a] + self.gamma*v1[:] ))
					# print (('Q[a =',a ,',s =', s,'] =',"%.2f" % Q[a]))
				v2[s] = np.max(Q)
				optimalAction[s] = np.argmax(Q)
			
			# twodecimals = ["%.2f" % var for var in v2]
			# print ((twodecimals))
			# print ((i , '->', twodecimals))
			# print ((np.linalg.norm(v2 - v1) ))
			# print ((optimalAction))
			# elapsed_time = time.time() - start_time
			# print ("loop time: %.5f" % elapsed_time)

		elapsed_time = time.time() - start_time
		print ("elapsed time: %.5f" % elapsed_time)
		return optimalAction


def main():
	# try:
	#     fileName = sys.argv[1]
	# except:
	#     fileName = input("Enter Filename: ")
	mdp = MDP()
	# optimalAction = mdp.valueIteration()
	# df_out = pd.DataFrame(data={'idstate': np.array(list(mdp.states))\
	# 	,'idaction': optimalAction}, columns=['idstate', 'idaction'])
	# print ((df_out.head()))
	# df_out.to_csv('out.csv')
	mdp.lstd()


if __name__ == "__main__": main()
# dist = numpy.linalg.norm(a-b)