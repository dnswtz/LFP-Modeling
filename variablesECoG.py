#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ! ==================================================================================== !
# !                                                                                      !
# !                                                                                      !
# !       Preliminary definition of the parameters that are to be used as part of        !
# !                the truConnect.py and ecog_testCorrect.py modules                     !
# !                                                                                      !
# !                                                                                      !
# ! ==================================================================================== !

"""
variablesECoG is designed in order to be imported into truConnect.py and ecog_testCorrect.py.
This is done with the aim to create a cleaner code by keeping each individual step of the simulation
seperate from the others:

	1. Definition of variables (variablesECoG.py)
	2. Creation of functions defining the connectivity parameters (truConnect.py)

		* interPop(adj_dict, numExc=excPerPop, numInh=inhPerPop, numPop=popSize

		* interElectrode_exc(adj_dict, extent, sigma, numInhNeurons=inhNeurons_tot,
			numNeurons=excNeurons_tot, excNrnsPPop = excPerPop, allow_autapse =
			False)

		* interElectrode_inh(adj_dict, extent, sigma, numExcNeurons=excNeurons_tot,
			numNeurons=inhNeurons_tot, inhNrnsPPop = inhPerPop, allow_autapse =
			False)

		* deviceConn(adj_dict, numExcNeurons = excNeurons_tot, numInhNeurons =
			inhNeurons_tot, totNodes=numNodes)

	3. Simulation is run (ecog_testCorrect.py)

[...]

"""

import numpy as np
import random

import nest
import nest.topology as tp

import sys
import time
import datetime
np.random.seed(12345)


dt = 0.1
simtime = 5000.

'''
Defines date and time. These parameters shall be used later whence naming the
files that keep the spike timings.
'''
starttime = datetime.datetime.now()
date = "%s%s%s"%(starttime.day,starttime.month,starttime.year)
time =  "%s%s%s"%(starttime.hour,starttime.minute,starttime.second)


'''
Defines connectivity parameters
'''

sigma_ex = 2.0
sigma_in = 0.1
sigma_LR = 5.

'''
Defines each node present in the simulation
'''

size_A1 = 32000
size_A2 = 32000

excNeurons = int(size_A1 * 0.8)
inhNeurons = int(size_A1 - excNeurons)

numNeurons = size_A1 + size_A2

excSpike = 1
inhSpike = 1

poissonGen = 1

numNodes = numNeurons + excSpike + inhSpike + poissonGen


out_degree_LV_ex = 600
out_degree_LV_in = 600


targets_created = False

xGrid = 5. #5 in cm
yGrid = 5. #5 in cm


'''
Characterization of parameters important for network dynamics
'''
eta = 2.0               # ratio of external to threshold rate of poisson input
g = 8.0                 # ration inhibitory to CEitatory weight


tauSyn = 2.0 #0.5               # synaptic time constant in ms
tauMem = 20.0           # time constant of membrane potential in ms
CMem = 250.0            # capacitance of membrane in in pF
theta = 20.0              # membrane threshold potential in mV
neuron_params = {"C_m": CMem,
		"tau_m": tauMem,
		"tau_syn_ex": tauSyn,
		"tau_syn_in": tauSyn,
		"t_ref": 2.0,
		"E_L": 0.0,
		"V_reset": 0.,
		"V_m": 0.0,
		"V_th": theta}

JE = 1. # J / J_unit    # amplitude of excitatory postsynaptic current
JI = -g * JE            # amplitude of inhibitory postsynaptic current


'''
Calculates the threshold value of the system and determines the rate
of the external nosie.
'''

p_rate = 45000.0

delay = 1.5             # inter population delay
delay_LR = 3.0          # inter electrode delay ->> include distance dependence


'''
Creates the random positioning for each neuron
'''
np.random.seed(1728937)

pos_A1 = [[np.random.uniform(-xGrid,0),np.random.uniform(-yGrid,yGrid)] for nrnID in range(size_A1)]
pos_A2 = [[np.random.uniform(-0,xGrid),np.random.uniform(-yGrid,yGrid)] for nrnID in range(size_A2)]


adj_A1 = {}
adj_A2 = {}
for neuron in range(size_A1):
        adj_A1.update({neuron:{'position':pos_A1[neuron], 'targets':[], 'delays':delay, 'horizontal_targets':[], 'devices':[], 'distance_to_electrode':[]}})
for neuron in range(size_A2):
	adj_A2.update({neuron:{'position':pos_A2[neuron], 'targets':[], 'delays':delay, 'horizontal_targets':[], 'devices':[], 'distance_to_electrode':[]}})

# MultiElectrodeArray parameters
sigma_cond = 0.3
potential_reach = 1.
samplingRate = 0.5




