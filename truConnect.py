#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ! ==================================================================================== !
# !                                                                                      !
# !                                                                                      !
# !       Creation of adjacency parameters that may be fed into the nest.Topology        !
# !                         defined in ecog_testCorrect.py                               !
# !                                                                                      !
# !                                                                                      !
# ! ==================================================================================== !

"""
truConnect is a program designed to create an adjacency parameters that establish
the connectivity between a large amount of neurons, a noise generator and their
corresponding spike detectors. The connectivity follows a constant probability
pattern for neurons within a population (inter populatory) and a gaussian
connectivity profile for neurons between electrodes == spike detectors (inter
electrode or long range connectivity).
"""
import variablesECoG as var

import nest
import nest.topology as tp

import numpy as np
import random
import math
import sys
import time
np.random.seed(12345)

connProfile = lambda x, sigma: 1./(sigma*np.sqrt(2*np.pi))*np.exp(-1./2*(x/sigma)**2)

def arealConn(adj_dict, inhSigma, excSigma, excSigma_LR, out_degree_ex = var.out_degree_LV_ex, out_degree_in = var.out_degree_LV_in, 
			out_degree_LR = var.out_degree_LR, numInhNeurons = var.inhNeurons, numExcNeurons = var.excNeurons, JI = var.JI, JE = var.JE,
			extent = [var.xGrid, var.yGrid], nodes = var.numNodes, allow_autapse = True, allow_multapse = 
			True, wrap_around = False):

	'''
	A gaussian long range connectivity profile is defined. Each neuron i defined in the adjacency matrix
	has a spatial coordinate around it which acts as the center of circle. This circle defines the extent of
	long range connetivity. Each neuron j within this circle is connected to i according to the specified
	connectivity parameters and the (i,j)th value in the adjecency matrix is set to 1. The ith dictionary
	entry specifying the 'target'ed neurons is updated by j.
	'''
	#percentage = 0

	print("\t*Creating inter areal connections")

	psnGen = nodes - 1
	popSize = numInhNeurons + numExcNeurons
	potConnect = np.arange(popSize)

	stored_targets = open('stored_targets.csv', 'a')
	target_distance = open('target_distance.csv', 'a')

	for nrn in range(popSize):

		pos = adj_dict[nrn]['position']
		connect2nrn = potConnect[:]
		distance = []

		for conNrn in range(popSize):

			conPos = adj_dict[conNrn]['position']

			if wrap_around:

				x_dis = abs(pos[0] - conPos[0])
				y_dis = abs(pos[1] - conPos[1])

				if (x_dis > extent[0]/4.):
					x_dis = extent[0]/2. - x_dis

				if (y_dis > extent[1]/2.):
					y_dis = extent[1] - y_dis


				distance.append(np.sqrt(x_dis**2 + y_dis**2))

			else:
				distance2nrn = np.sqrt((pos[0] - conPos[0])**2 + (pos[1] - conPos[1])**2)
				distance.append(distance2nrn)

		distance = np.array(distance)

		if not allow_autapse:
			autapse = np.where( connect2nrn == nrn )
			np.delete(connect2nrn, autapse)
			np.delete(distance, autapse)

		if nrn < 25600:

			conProb = connProfile(distance, excSigma)

			excNorm = np.sum(conProb)
			conProb = conProb/excNorm
			targets = list(np.random.choice(a = connect2nrn, size = out_degree_ex, replace=allow_multapse, p = conProb))

		if nrn >= 25600:
			conProb = connProfile(distance, inhSigma)

			inhNorm = np.sum(conProb)
			conProb = conProb/inhNorm
			targets = list(np.random.choice(a = connect2nrn, size = out_degree_in, replace=allow_multapse, p = conProb))
			

		#conProb = np.append(conProb_ex, conProb_exLR)
		#conProb = np.append(conProb, conProb_in)		

		#excNormalization = np.sum(conProb[:numExcNeurons])
		#excLRNormalization = np.sum(conProb[numExcNeurons:2*numExcNeurons])
                #inhNormalization = np.sum(conProb[2*numExcNeurons:])

                #conProb[:numExcNeurons] = conProb[:numExcNeurons]/excNormalization
		#conProb[numExcNeurons:2*numExcNeurons] = conProb[numExcNeurons:2*numExcNeurons]/excLRNormalization
                #conProb[2*numExcNeurons:] = conProb[2*numExcNeurons:]/inhNormalization


                #excTargets = list(np.random.choice(a = connect2nrn[:numExcNeurons], size = out_degree_ex, replace=allow_multapse, p = conProb[:numExcNeurons]))
		#excLRTargets = list(np.random.choice(a = connect2nrn[:numExcNeurons], size = out_degree_LR, replace=allow_multapse, p = conProb[numExcNeurons:2*numExcNeurons]))
                #inhTargets = list(np.random.choice(a = connect2nrn[numExcNeurons:], size = out_degree_in, replace=allow_multapse, p = conProb[2*numExcNeurons:]))

		#targets = np.concatenate((excTargets, excLRTargets, inhTargets))
		#print(targets)

		#if not allow_autapse:
		#	autapse = np.where( targets == nrn )
		#	np.delete(targets, autapse)

		for target in targets:

			adj_dict[nrn]['targets'].append(target)
			stored_targets.write(str(target) + ',')

		stored_targets.write('\n')

		distance = distance[(targets,)]
		for eucl in distance:
			target_distance.write(str(eucl)+',')

		

		target_distance.write('\n')

	target_distance.close()
	stored_targets.close()

def horizontalConn(adj_dict, adj_dict_target, numExcNeurons = var.excNeurons, numNeurons = var.size_A1, out_degree = var.out_degree_LR):

	print("\t*Creating horizontal connections")

	target_distance = open('target_distance.csv', 'a')

	for neuronID in range(numExcNeurons):
		horizontal_targets = np.random.randint(0, numNeurons, out_degree)
		horizontal_targets = horizontal_targets.tolist()

		adj_dict[neuronID]['horizontal_targets'] = horizontal_targets

		posID = adj_dict[neuronID]['position']

		for target in horizontal_targets:
			posTarget = adj_dict_target[target]['position']
			distance = np.sqrt((posTarget[0] - posID[0])**2 + (posTarget[1] - posID[1])**2)
			target_distance.write(str(distance)+',')

		target_distance.write('\n')

	target_distance.close()

def deviceConn(adj_dict, numExcNeurons = var.excNeurons, numInhNeurons = var.inhNeurons, totNodes = var.numNodes):
	'''
	Connection of all excitatory and inhibitory neurons to their corresponding
	spike train detectors. This is once again done by changing the corresponding
	value to 1 in the adjacency matrix. Analogously, the poisson generator is
	connected to each neuron.
	'''

	print("\t*Connecting devices")

	spk_exc = totNodes - 3
	spk_inh = totNodes - 2

	# Connection of excitatory neurons to excitatory spike detector:
	[adj_dict[excNrn]['devices'].append(spk_exc) for excNrn in range(numExcNeurons)]
	# Connection of inhibitory neurons to inhibitory spike detector:
	[adj_dict[inhNrn]['devices'].append(spk_inh) for inhNrn in range(numExcNeurons, numExcNeurons + numInhNeurons)]


def layerNet(pos_A1 = var.pos_A1, pos_A2 = var.pos_A2, xGrid = var.xGrid, yGrid = var.yGrid):
	'''
	This function has been created, as arealConn( *kwargs ) was designed to connect
	the populations, excitatory- and inhibitory recorder and lastly the poisson generator
	into account, in that order. By creating calling layerNet() no additional bugs will
	be introduced into the code.
	'''

	print("\t*Populating environment")

	pop_A1 = tp.CreateLayer({'positions': pos_A1, 'extent':[float(xGrid+10),float(yGrid+10)], 'elements': 'iaf_psc_alpha', 'edge_wrap': True})
	pop_A2 = tp.CreateLayer({'positions': pos_A2, 'extent':[float(xGrid+10),float(yGrid+10)], 'elements': 'iaf_psc_alpha', 'edge_wrap': True})

	print("\t*Creating recording devices")

	excRec = nest.Create('spike_detector')
	inhRec = nest.Create('spike_detector')

	print("\t*Creating noise generator")

	noise = nest.Create('poisson_generator')

	return pop_A1, pop_A2, excRec, inhRec, noise


def main():

	if not var.targets_created:
		arealConn(adj_dict = var.adj_A1, excSigma = var.sigma_ex, inhSigma = var.sigma_in, excSigma_LR = var.sigma_LR)
		arealConn(adj_dict = var.adj_A2, excSigma = var.sigma_ex, inhSigma = var.sigma_in, excSigma_LR = var.sigma_LR)

	#horizontalConn(adj_dict = var.adj_A1, adj_dict_target = var.adj_A2)
	#horizontalConn(adj_dict = var.adj_A2, adj_dict_target = var.adj_A1)

	deviceConn(adj_dict = var.adj_A1)
	deviceConn(adj_dict = var.adj_A2)

if __name__ == "__main__":
	main()
