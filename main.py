#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ! ==================================================================================== !
# ! 											 !
# !  											 !
# ! 		Simulation of ECoG data in a nest.topology environment			 !
# !											 !
# !											 !
# ! ==================================================================================== !

from __future__ import print_function

import lfpcalc as lfp
import variablesECoG as var
import truConnect as tools

import scipy
import numpy as np
import math

import nest
import nest.topology as tp

import time
import datetime
import sys

np.random.seed(12345)

nest.ResetKernel()
nest.SetKernelStatus({"local_num_threads": 8, "resolution": var.dt, "print_time": True, "overwrite_files": True})


'''
layerNet is a list designated for the inclusion of #popSize different layers (populations)
and their poisson input. The individual layers are created in a loop of length corresponding
to #popSize and appended to layerNet in a tuple containing the composite element excitatory
neuron and excitatory noise and inhibitory neuron and inhibitory noise separately. The
command exPop = ()/ inPop = () is called in order to clear the variables memory [sort of like
a low budget nest.ResetKernel() ] after each iteration.
'''

print("Preparing Defaults")

nest.SetDefaults('iaf_psc_alpha', var.neuron_params)
nest.SetDefaults("poisson_generator", {"rate": var.p_rate})


'''
Defines connection dictionary for short- and long term connectivity. Furthermore, defaults
for the neurons and poisson generator used are customized.
'''

nest.CopyModel("static_synapse", "spike_det", {"weight": var.JE, "delay": var.delay})

'''
Creates a spike detector layer. Each population is read out simultaneously and added to the
file - ::file:: - whilst reading out note the neural order.
'''

pop_A1, pop_A2, excRec, inhRec, noise = tools.layerNet()

nest.SetStatus(excRec, [{'label': "Excitatory-" + var.date + "_" + var.time, "withtime":True, "withgid":True, "to_file":True}])
nest.SetStatus(inhRec, [{'label': "Inhibitory-" + var.date + "_" + var.time, "withtime":True, "withgid":True, "to_file":True}])

print("Finished Preparing Defaults")

'''
The nodes from each topology layer are read out using nest.GetNodes, the resulting list
is used to establish a connection between neurons given the adjacency dictionary created
in truConnect
'''

A1_nodes = nest.GetNodes(pop_A1)[0]
A2_nodes = nest.GetNodes(pop_A2)[0]

# note that the order of the individual nodes in the list 'nodes' is deliberate
nodes = A1_nodes + A2_nodes + excRec + inhRec + noise


print("Preparing ECoG electrodes")

all_neuronal_nodes = []
all_neuronal_nodes.extend(A1_nodes)
all_neuronal_nodes.extend(A2_nodes)
all_neuronal_nodes = tuple(all_neuronal_nodes)

electrodes = lfp.MEA(True)
electrodes.create_virtual_electrode(all_neuronal_nodes)

print("Finished preparing ECoG electrodes")


'''
As each leach consists of a composite element of excitatory neuron/excitatory noise or
inhibitory neuron/inhibitory noise, we can simpy connect the nodes with themselves, whilst
specifying the source and target in the connection dictionary - noiseConn - excitatory
and inhibitory noise to population connections is treated equally.
'''

print("Connecting network")

tools.main()


print('Connecting Network')
for nrn in var.adj_A1.items():
	neuronID = nrn[0]
	parameter = nrn[1]
	print(nrn[0])

	# connects poisson generator to target neuron 'nrn'
	nest.Connect(noise, [A1_nodes[neuronID]], "one_to_one", {'weight':var.JE, 'delay':var.delay})

	# connects target neuron 'nrn' to a spike detector
	if neuronID < var.size_A1:
		nest.Connect([A1_nodes[neuronID]], [nodes[parameter['devices'][0]]], syn_spec = "spike_det")

	# connects neurons within an areal
	if not var.targets_created:
		for target in parameter['targets']:
			if neuronID < var.excNeurons:
				weight = var.JE
			else:
				weight = var.JI
			conn_dict = {"weight":weight, "delay":var.delay}
			nest.Connect([A1_nodes[neuronID]], [A1_nodes[target]], "one_to_one", conn_dict)

	# connects neurons from outside the local volume
	#conn_dict_hztl = {"weight":var.JE, "delay":var.delay_LR}

	#for target in parameter['horizontal_targets']:
	#	nest.Connect([A1_nodes[neuronID]], [A2_nodes[target]], "one_to_one", conn_dict_hztl)


for nrn in var.adj_A2.items():
	neuronID = nrn[0]
	parameter = nrn[1]
	print(nrn[0])
	# connects poisson generator to target neuron 'nrn'
	nest.Connect(noise, [A2_nodes[neuronID]], "one_to_one", {'weight':var.JE, 'delay':var.delay})

	# connects target neuron 'nrn' to a spike detector
	if neuronID < var.size_A2:
		nest.Connect([A2_nodes[neuronID]], [nodes[parameter['devices'][0]]], syn_spec = "spike_det")

	# connects neurons within an areal
	if not var.targets_created:
		for target in parameter['targets']:
			if neuronID < var.excNeurons:
				weight = var.JE
			else:
				weight = var.JI
			conn_dict = {"weight":weight, "delay":var.delay}
			nest.Connect([A2_nodes[neuronID]], [A2_nodes[target]], "one_to_one", conn_dict)

	# connects neurons from outside the local volume
	#conn_dict_hztl = {"weight":var.JE, "delay":var.delay_LR}

	#for target in parameter['horizontal_targets']:
	#	nest.Connect([A2_nodes[neuronID]], [A1_nodes[target]], "one_to_one", conn_dict_hztl)


if var.targets_created:
	with open('stored_targets.csv', 'r') as stored_targets:
		nrn = 0
		for targets in stored_targets:
			print(nrn)
			targets = targets.strip()
			targets = targets.split(',')
			del targets[-1]

			if (nrn < var.excNeurons) or (var.size_A1 <= nrn < var.size_A1 + var.excNeurons):
				weight = var.JE
			else:
				weight = var.JI

			conn_dict = {"weight":weight, "delay":var.delay}
			if nrn < var.size_A1:
				[nest.Connect([nodes[nrn]], [nodes[int(target)]],  "one_to_one", conn_dict) for target in targets]
			else:
				[nest.Connect([nodes[nrn]], [nodes[int(var.size_A1) + int(target)]],  "one_to_one", conn_dict) for target in targets]
			nrn+=1

print("Finished Network Connections")

print("Simulating")
nest.Simulate(var.simtime)

electrodes.create_mea_data()
#electrodes.simulate_potential()
print('Potential simulated')
