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

'''
Function describing the percentage of a task that has already been completed
'''

def percentageComplete(perc):
	p = int(round(perc))
	sys.stdout.write(('[')+('='*p)+(' '*(100-p))+(']')+("\r [ %d"%p+"% ]"))
	sys.stdout.flush()

'''
Definition of functions used in this example. First, define the
Lambert W function implemented in SLI. The second function
computes the maximum of the postsynaptic potential for a synaptic
input current of unit amplitude (1 pA) using the Lambert W
function. Thus function will later be used to calibrate the synaptic
weights.
'''

def LambertWm1(x):
	nest.sli_push(x)
	nest.sli_run('LambertWm1')
	y = nest.sli_pop()
	return y


def ComputePSPnorm(tauMem, CMem, tauSyn):
	a = (tauMem / tauSyn)
	b = (1.0 / tauSyn - 1.0 / tauMem)

	# time of maximum
	t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)

	# maximum of PSP for current of unit amplitude
	return (np.exp(1.0) / (tauSyn * CMem * b) * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b -
		t_max * np.exp(-t_max / tauSyn)))



'''
Assigning the current time to a variable in order to determine the
build time of the network.
'''

#startbuild = time.time()

dt = 0.1
simtime = 7000.

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

pConn = 0.1
sigma_ex = 330*10**(-4)
sigma_in = 130*10**(-4)
sigma_LR = 5.0

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


#out_degree = 230 #int(size_A1*0.1)           	#total number of outputs from a neuron

out_degree_LV_ex = 400  #int(out_degree*0.45)		#total number of connections within local volume (>500um)
out_degree_LV_in = 100
out_degree_LR = 175 #int(out_degree*0.55*0.7)	#total number of long range connections from other areal


targets_created = True

'''
Characterization of parameters important for grid layour of the neuronal
layers. Can be chosen arbitrarily, the only constraint is:

        xGrid * yGrid = popSize

Other than that the network size should work for any given layout.
'''

xGrid = 5. #5 in cm
yGrid = 5. #5 in cm


'''
Characterization of parameters important for network dynamics
'''
eta = 2.0               # ratio of external to threshold rate of poisson input
g = 8.0                 # ration inhibitory to CEitatory weight

CE = pConn * excNeurons
CI = pConn * inhNeurons

'''
Initialization of the parameters of the integrate and fire neuron and
the synapses. The parameter of the neuron are stored in a dictionary.
The synaptic currents are normalized such that the amplitude of the
PSP is J.
'''

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

# find a way to do this with numpy.random.rand() and then use tolist() -> example: numpy.random.uniform(low,high, size)
pos_A1 = [[np.random.uniform(-xGrid,0),np.random.uniform(-yGrid,yGrid)] for nrnID in range(size_A1)]
pos_A2 = [[np.random.uniform(-0,xGrid),np.random.uniform(-yGrid,yGrid)] for nrnID in range(size_A2)]


'''
Creates the connection dictionary, in which the connectivity
from node i to node j is defined, the connection weight and delay, as well as the
position of each node and the devices the node is conencted to are described.
'''

adj_A1 = {}
adj_A2 = {}
for neuron in range(size_A1):
        adj_A1.update({neuron:{'position':pos_A1[neuron], 'targets':[], 'delays':delay, 'horizontal_targets':[], 'devices':[], 'distance_to_electrode':[]}})
for neuron in range(size_A2):
	adj_A2.update({neuron:{'position':pos_A2[neuron], 'targets':[], 'delays':delay, 'horizontal_targets':[], 'devices':[], 'distance_to_electrode':[]}})

# MultiElectrodeArray parameters
sigma_cond = 0.3
potential_reach = 1.
samplingRate = 1.




