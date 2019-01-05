from __future__ import print_function

import numpy as np
import pandas as pd

import nest
import h5py
import random
import time

import scipy.stats
import variablesECoG as var


class MEA:

	def __init__(self, top_left, *positions):

		self.xGrid = var.xGrid
		self.yGrid = var.yGrid
		self.potential_reach = var.potential_reach
		
		self.neural_posX = np.array([])
		self.neural_posY = np.array([])

		if top_left:
			x = np.linspace(-self.xGrid+self.potential_reach, self.xGrid-self.potential_reach,8)
			y = np.linspace(-self.yGrid+self.potential_reach, self.yGrid-self.potential_reach,8)
			self.positions = tuple([[xx,yy] for xx in x for yy in y])
		else:
			self.positions = positions ##### problem?

	def create_virtual_electrode(self, all_neuronal_nodes, samplingRate=var.samplingRate):

		self.virt_electrodes = nest.Create('multimeter', params = {'withtime':False, 'interval':samplingRate, 'withgid':True, 'record_from': ['I_syn_ex', 'I_syn_in']})

		nest.Connect(self.virt_electrodes, all_neuronal_nodes)

	def sort_rank(self, a, step): # same as scipy.stats.rankdata(a, method='ordinal')

		'''
		self.sort_rank(a) is equivalent to scipy.stats.rankdata(a, method='original') in functionality. Had to be reimplemented though
		on account of compatibility issues.

		--------------------------------------------------------
		Parameters:

			a: 	list of sender IDs from multimeter
			step: 	sample length of the multimeter. Note
				due to the way NEST measures currents
				the sample length is equivalent to
				simulation time * samplingRate - 1

		Returns:

			mask: 	list of ranks based on the IDs occurance
				in a

		--------------------------------------------------------
		'''

		start = 0
		mask = np.argsort(a)
		for jj in range(len(a)):
			mask[start:start+step] = np.sort(mask[start:start+step])
			start+=step

		return mask

	def create_mea_data(self, adj_A1 = var.adj_A1, adj_A2 = var.adj_A2, nodes=int(var.numNeurons)):

		neural_posX = self.neural_posX
		neural_posY = self.neural_posY

		for neuron in adj_A1:
			print(neuron)
			pos = np.asarray(adj_A1[neuron]['position'])
			neural_posX = np.concatenate((neural_posX,[pos[0]]))
			neural_posY = np.concatenate((neural_posY,[pos[1]]))
			[adj_A1[neuron]['distance_to_electrode'].append(np.linalg.norm(pos-np.asarray(electrode_pos))) for electrode_pos in self.positions]

		for neuron in adj_A2:
			print(neuron)
			pos = np.asarray(adj_A2[neuron]['position'])
			neural_posX = np.concatenate((neural_posX,[pos[0]]))
			neural_posY = np.concatenate((neural_posY,[pos[1]]))			
			[adj_A2[neuron]['distance_to_electrode'].append(np.linalg.norm(pos-np.asarray(electrode_pos))) for electrode_pos in self.positions]

		with h5py.File('currents.hdf5', 'w') as mea:

			print('Prepping virtGID')
			virtGID = nest.GetStatus(self.virt_electrodes)[0]['events']['senders']
			id1 = virtGID[0]
			self.samples = len(np.where( virtGID == id1 )[0])

			virtGID = pd.Series(virtGID)
			mask = pd.Series.rank(virtGID, method='first').astype(int)-1
			del virtGID
			mask = np.array(mask)
			GID_sorted = np.argsort(mask)
			del mask

			print('Prepping Synaptic Currents')
			syn_currents_ex = nest.GetStatus(self.virt_electrodes)[0]['events']['I_syn_ex']
			syn_currents_in = nest.GetStatus(self.virt_electrodes)[0]['events']['I_syn_in']

			print('Summing Synaptic Currents')
			currents_sum = np.array(map(sum, zip(*[syn_currents_ex,syn_currents_in])))
			del syn_currents_ex
			del syn_currents_in

			print('Saving Currents')
			virt_currents = currents_sum[GID_sorted].reshape(nodes,self.samples)
			del GID_sorted
			mea.create_dataset('currents', data=virt_currents)
			del virt_currents

			for ecog_electrode in range(len(self.positions)):

				print(ecog_electrode)
				distance_A1 = np.array([adj_A1[nrn]['distance_to_electrode'][ecog_electrode] for nrn in adj_A1])
				distance_A2 = np.array([adj_A2[nrn]['distance_to_electrode'][ecog_electrode] for nrn in adj_A2])

				distances = np.concatenate((distance_A1, distance_A2), axis=0)

				mea.create_dataset('ecog_distance:%d'%ecog_electrode, data=distances)

			del distances

	def potential(self, sigma, In, r):
		return  1./(4*np.pi*sigma)*In/(r*0.01)

	def simulate_potential(self, sigma=var.sigma_cond, adj_A1=var.adj_A1, adj_A2=var.adj_A2): # find work around -> one dictionary at most
		potentials = h5py.File('ecog_potential.hdf5', 'w')

		for ecog_electrode in range(len(self.positions)):
			print('Currently calculating the potential of electrode %d'%ecog_electrode)
			with h5py.File('currents.hdf5', 'r') as mea:

				ecog_distance = mea['ecog_distance:%d'%ecog_electrode]
				ecog_distance = ecog_distance[...]

				currents = mea['currents']
				currents = currents[...].T

			phi_contrib = np.array([self.potential(sigma, currents[time], ecog_distance) for time in range(len(currents))])
			phi = phi_contrib.sum(axis=1)
			potentials.create_dataset('potential_at_ecog%d'%ecog_electrode, data=phi)

		potentials.close()

	def no_volume_conduction(self, sigma=var.sigma_cond, adj_A1=var.adj_A1, adj_A2=var.adj_A2):
		
		neural_posX = self.neural_posX
		neural_posY = self.neural_posY

		potentials = h5py.File('noVolumeConduction.hdf5', 'w')

		for ecog_electrode in range(len(self.positions)):
			print('Currently calculating the potential of electrode %d'%ecog_electrode)
			x = self.positions[ecog_electrode][0]
			y = self.positions[ecog_electrode][1]

			x_reach = np.where((x-1.14289 < neural_posX)&(neural_posX <= x+1.14289))[0] #neuronID within x-1.14 and x+1.14
			y_reach = np.where((x-1.14289 < neural_posY)&(neural_posY <= x+1.14289))[0] #neuronID within y-1.14 and y+1.14

			print(x_reach)

			reach = np.where(x_reach == y_reach) # neuronID within reach of electrode (volume conduction neglected)

			with h5py.File('currents.hdf5', 'r') as mea:

				ecog_distance = mea['ecog_distance:%d'%ecog_electrode][...]
				ecog_distance = ecog_distance[reach]
				
				currents = mea['currents'][...]
				currents = currents[reach].T

			phi_contrib = np.array([self.potential(sigma, currents[time], ecog_distance) for time in range(len(currents))])
			phi = phi_contrib.sum(axis=1)
			potentials.create_dataset('potential_at_ecog%d'%ecog_electrode, data=phi)

		potentials.close()
