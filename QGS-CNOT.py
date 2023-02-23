#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:37:31 2023

@author: oqi
"""
import os
# disable GPU due to memory constraints
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from QGS import *

import matplotlib.pyplot as plt
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Ket, BSgate, Rgate, Interferometer, MZgate
from strawberryfields.utils import operation

# Permutation matrix
SWAP = np.array([[0, 1], [1, 0]])

#%% 4 Photons on 8 modes
# input state:
initial_state = [(1,0,1,0,1,0,1,0),
                 (1,0,1,0,1,0,0,1),
                 (0,1,1,0,1,0,1,0),
                 (0,1,1,0,1,0,0,1)]

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

post_select = [[1,0,1,0],[0,1,None,None,None,None,2,3]]

@operation(8)
def Li_et_al_2021(q):   
    
    # Preperation of Ancilla states
    BSgate() | (q[2], q[3])
    
    # PBS1
    Interferometer(SWAP) | (q[0], q[2])
    
    #PBS2
    BSgate() | (q[4], q[5])
    BSgate() | (q[6], q[7])
    Interferometer(SWAP) |(q[4], q[6])
    BSgate().H | (q[4], q[5])
    BSgate().H | (q[6], q[7])

    
    #PBS3
    Rgate(np.pi/2) | q[2]
    BSgate() | (q[2], q[3])
    #Interferometer(SWAP) | (q[0], q[1])
    Rgate(-np.pi/2) | q[2]
    Rgate(-np.pi/2) | q[3]
    
    Rgate(np.pi/2) | q[4]
    BSgate() | (q[4], q[5])
    #Interferometer(SWAP) | (q[2], q[3])
    Rgate(-np.pi/2) | q[4]
    Rgate(-np.pi/2) | q[5]
    
    Interferometer(SWAP) | (q[2], q[4])
    
    Rgate(np.pi/2) | q[4]
    Rgate(np.pi/2) | q[5]
    BSgate().H | (q[4], q[5])
    Rgate(-np.pi/2) | q[4]
    
    Rgate(np.pi/2) | q[2]
    Rgate(np.pi/2) | q[3]
    BSgate().H | (q[2], q[3])
    Rgate(-np.pi/2) | q[2]
    

qgs = QGS(4, initial_state, layers=1, modes=8)
qgs.fit(target_state, fail_states = fail_state, reps = 5000, path = '4p8m/4p8m_prep_OLD.npz', cost_factor = 1, post_select = post_select, p_success=None, preparation=Li_et_al_2021)
qgs.evaluate(target_state, post_select=post_select, verbosity = 1, preparation=Li_et_al_2021)
#%%
ps, pss = AncillaStates(2, 4)
ket = None
for i in range(len(ps)):
    print(ps[i])
    post_select = [ps[i],[0,1,2,3,None,None,None,None]]
    
    ket = qgs.evaluate(target_state, fail_state, post_select=post_select, title = '4p8m with 10 layers - Measure: ' + pss[i], path = '4p8m/4p8m_l10_'+pss[i], ket = ket, return_ket = True)

#%% 6 Photons on 10 modes - sufficient for EPR generation
# input state:
initial_state = [(1,0,1,0,1,1,1,1,0,0),
                 (1,0,0,1,1,1,1,1,0,0),
                 (0,1,1,0,1,1,1,1,0,0),
                 (0,1,0,1,1,1,1,1,0,0)]

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

qgs = QGS(4, initial_state, layers=1, decomposition='Reck', modes=9)
qgs.fit(target_state, fail_states = fail_state, reps = 1000, n_sweeps=7, path = '6p9m', sweep_low = 0.232, sweep_high = 0.25)

#%% 2 Photons on 6 modes, p_success = 1/9
# input state:
initial_state = [(0,1,0,1,0,0),
                 (0,1,0,0,1,0),
                 (0,0,1,1,0,0),
                 (0,0,1,0,1,0)]

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

post_select  = [[0,0],[None,0,1,2,3,None]]

sweep = np.linspace(0.01, 0.25, 25)

qgs = QGS(2, initial_state, layers=1, modes=6)
qgs.fit(target_state, fail_states = fail_state, post_select=post_select, reps=1000, load = 'Test/2p6m.npz', path = '2p6m/free_opt2.npz', punish=0.0, cost_factor=0.001, norm=[2,2,'mean'])
qgs.evaluate(target_state, fail_states = fail_state, path = '2p6m/free_opt2', post_select=post_select)

#%% 3 Photons on 6 modes, p_success = 1/8
# input state:
initial_state = [(1,0,1,0,1,0),
                 (1,0,1,0,0,1),
                 (0,1,1,0,1,0),
                 (0,1,1,0,0,1)]

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

post_select  = [[1,0],[0,1,None,None,2,3]]

qgs = QGS(3, initial_state, layers=2, modes=6)
qgs.fit(target_state, fail_states = fail_state, post_select=post_select, reps=500, load = 'Test/3p6m.npz', path = '3p6m/free_opt.npz', norm=[2,np.inf,'mean'], punish=0.0001, cost_factor=0.001)
qgs.evaluate(target_state, fail_states = fail_state, path = '3p6m/free_opt', post_select=post_select)

#%% Liu and Wei 2022

# input state:
initial_state = [(1,0,1,0,1,0),
                 (1,0,1,0,0,1),
                 (0,1,1,0,1,0),
                 (0,1,1,0,0,1)]

initial_state = FockBasis(initial_state, 4, 6)

# output state:
target_state =  [(1,0,1,0,1,0),
                 (1,0,1,0,0,1),
                 (0,1,1,0,0,1),
                 (0,1,1,0,1,0)]
# failed states:
fail_state = [(1,1,1,0,0,0),
              (0,0,1,0,1,1),
              (2,0,1,0,0,0),
              (0,2,1,0,0,0),
              (0,0,1,0,2,0),
              (0,0,1,0,0,2)]

LiuWei2022 = sf.Program(6)

with LiuWei2022.context as q:
    Ket(initial_state)  | q
    
    # Prepare Ancilla in the state 1/sqrt(2) (|H> + |V>)
    BSgate().H          | (q[2], q[3])
    Rgate(np.pi)        | q[3]    
    
    # Hadarmad Gate
    BSgate()            | (q[4], q[5])
    Rgate(np.pi)        | q[5]
    
    # CZ Protocoll
    Interferometer(SWAP) | (q[0], q[2])  # PBS1 transmitt H refelect V
    BSgate().H          | (q[2], q[3])  # HWP1 @ 22.5°
    Rgate(np.pi)        | q[3]
    Interferometer(SWAP) | (q[2], q[4])  # PBS2 transmitt H reflect V
    BSgate().H          | (q[2], q[3])  # HWP2 @ 22.5°
    Rgate(np.pi)        | q[3]
    
    # Hadarmad Gate
    BSgate()            | (q[4], q[5])
    Rgate(np.pi)        | q[5]

evaluate(LiuWei2022, target_state, fail_state, cutoff_dim=4, title = 'Liu and Wei 2022', path=("Test/LiuWei_2022"))

#%% Liu and Wei 2022 with feedforward

# input state:
initial_state = [(1,0,1,0,1,0),
                 (1,0,1,0,0,1),
                 (0,1,1,0,1,0),
                 (0,1,1,0,0,1)]

initial_state = FockBasis(initial_state, 4, 6)

# output state:
target_state =  [(1,0,0,1,1,0),
                 (1,0,0,1,0,1),
                 (0,1,0,1,0,1),
                 (0,1,0,1,1,0)]
# failed states:
fail_state = [(1,1,0,1,0,0),
              (0,0,0,1,1,1),
              (2,0,0,1,0,0),
              (0,2,0,1,0,0),
              (0,0,0,1,2,0),
              (0,0,0,1,0,2)]

LiuWei2022 = sf.Program(6)

with LiuWei2022.context as q:
    Ket(initial_state)  | q
    
    # Prepare Ancilla in the state 1/sqrt(2) (|H> + |V>)
    BSgate().H          | (q[2], q[3])
    Rgate(np.pi)        | q[3]    
    
    # Hadarmad Gate
    BSgate()            | (q[4], q[5])
    Rgate(np.pi)        | q[5]
    
    # CZ Protocoll
    Interferometer(SWAP) | (q[0], q[2])  # PBS1 transmitt H refelect V
    BSgate().H          | (q[2], q[3])  # HWP1 @ 22.5°
    Rgate(np.pi)        | q[3]
    Interferometer(SWAP) | (q[2], q[4])  # PBS2 transmitt H reflect V
    BSgate().H          | (q[2], q[3])  # HWP2 @ 22.5°
    Rgate(np.pi)        | q[3]
    Rgate(np.pi)        | q[5]          # Feedforward correction
    
    # Hadarmad Gate
    BSgate()            | (q[4], q[5])
    Rgate(np.pi)        | q[5]
    
    

evaluate(LiuWei2022, target_state, fail_state, cutoff_dim=4, title='Liu and Wei 2022 - Feedforward', path='Test/LiuWei2022_feedforward')

#%% Ralph et. al. 2002

# input state:
initial_state = [(0,1,0,1,0,0),
                 (0,1,0,0,1,0),
                 (0,0,1,1,0,0),
                 (0,0,1,0,1,0)]

initial_state = FockBasis(initial_state, 3, 6)

# output state:
target_state =  [(0,1,0,1,0,0),
                 (0,1,0,0,1,0),
                 (0,0,1,0,1,0),
                 (0,0,1,1,0,0)]
# failed states:
fail_state = [(0,1,1,0,0,0),
              (0,0,0,1,1,0),
              (0,2,0,0,0,0),
              (0,0,2,0,0,0),
              (0,0,0,2,0,0),
              (0,0,0,0,2,0)]

Ralph2002 = sf.Program(6)

with Ralph2002.context as q:
    Ket(initial_state)              | q
    
    # CNOT Prototcol:
    Rgate(np.pi)                    | q[3]          # B3 nu = 1/2, phase change upon refelction on top
    BSgate(np.pi/4, np.pi)          | (q[3], q[4])
    BSgate(np.arccos(np.sqrt(1/3))) | (q[0], q[1])  # B1 nu = 1/3
    BSgate(np.arccos(np.sqrt(1/3))) | (q[2], q[3])  # B2 nu = 1/3
    BSgate(np.arccos(np.sqrt(1/3))) | (q[4], q[5])  # B5 nu = 1/3
    Rgate(np.pi)                    | q[3]          # B4 nu = 1/2, phase change upon refelction on top
    BSgate(np.pi/4, np.pi)          | (q[3], q[4])
    

evaluate(Ralph2002, target_state, fail_state, title='Ralph et al. 2002', path = 'Test/Ralph')

#%% Li et al. 2021

initial_state = [(1,0,1,0,1,0,1,0),
                 (1,0,1,0,1,0,0,1),
                 (0,1,1,0,1,0,1,0),
                 (0,1,1,0,1,0,0,1)]

initial_state = FockBasis(initial_state, 4, 8)

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

ps, pss = AncillaStates(2, 4)

for i in range(len(ps)):
    print(ps[i])
    post_select = [ps[i],[0,1,None,None,None,None,2,3]]
    
    Li2021 = sf.Program(8)
    
    with Li2021.context as q:
        Ket(initial_state)  | q
        
        # Preperation of Ancilla states
        BSgate() | (q[2], q[3])
        
        # PBS1
        Interferometer(SWAP) | (q[0], q[2])
        
        #PBS2
        BSgate() | (q[4], q[5])
        BSgate() | (q[6], q[7])
        Interferometer(SWAP) |(q[4], q[6])
        BSgate().H | (q[4], q[5])
        BSgate().H | (q[6], q[7])
    
    
        #PBS3
        Rgate(np.pi/2) | q[2]
        BSgate() | (q[2], q[3])
        #Interferometer(SWAP) | (q[0], q[1])
        Rgate(-np.pi/2) | q[2]
        Rgate(-np.pi/2) | q[3]
        
        Rgate(np.pi/2) | q[4]
        BSgate() | (q[4], q[5])
        #Interferometer(SWAP) | (q[2], q[3])
        Rgate(-np.pi/2) | q[4]
        Rgate(-np.pi/2) | q[5]
        
        Interferometer(SWAP) | (q[2], q[4])
        
        Rgate(np.pi/2) | q[4]
        Rgate(np.pi/2) | q[5]
        BSgate().H | (q[4], q[5])
        Rgate(-np.pi/2) | q[4]
        
        Rgate(np.pi/2) | q[2]
        Rgate(np.pi/2) | q[3]
        BSgate().H | (q[2], q[3])
        Rgate(-np.pi/2) | q[2]
                
    
    evaluate(Li2021, target_state, fail_state, post_select=post_select,cutoff_dim=4, title = 'Li et al. 2021 - Measure: ' + pss[i], path = 'Test/Li2021_'+pss[i])
#%% 4 Photons on 8 modes p_success = 1/16
# input state:
initial_state = [(1,0,1,0,1,0,1,0),
                 (1,0,1,0,1,0,0,1),
                 (0,1,1,0,1,0,1,0),
                 (0,1,1,0,1,0,0,1)]

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

post_select  = [[[1,0,1,0],[0,1,0,1]],[0,1,None,None,None,None,2,3], True]

sweep = np.round(np.geomspace(1/16,1/4,25),4)

qgs = QGS(4, initial_state, layers=2, modes=8)
qgs.fit(target_state, fail_states = fail_state, post_select=post_select, reps=5000, load = 'Test/4p8m_multiple.npz', norm=[2,2,2], path = '4p8m_v2', punish=1, n_sweeps = sweep)
#qgs.evaluate(target_state, fail_states = fail_state, path = '4p8m/free_opt3', post_select=post_select)

#%%
ps, pss = AncillaStates(2, 4)
ket = None
for i in range(len(ps)):
    print(ps[i])
    post_select = [ps[i],[0,1,None,None,None,None,2,3]]
    
    ket = qgs.evaluate(target_state, fail_state, post_select=post_select, title = '4p8m with 2 layers - Measure: ' + pss[i], path = '4p8m/4p8m_free_opt_'+pss[i], ket = ket, return_ket = True)



#%% Adapted Li

initial_state = [(1,0,1,0,1,0,1,1,1,0),
                 (1,0,1,0,1,0,1,1,0,1),
                 (0,1,1,0,1,0,1,1,1,0),
                 (0,1,1,0,1,0,1,1,0,1)]

initial_state = FockBasis(initial_state, 6, 10)

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

Li_mod = sf.Program(10)

with Li_mod.context as q:
    Ket(initial_state)  | q
    
    # Preperation of Ancilla states
    BSgate() | (q[2], q[3])
    BSgate() | (q[6], q[7])
    
    # PBS1
    Interferometer(SWAP) | (q[1], q[3])
    
    #PBS2
    BSgate() | (q[4], q[5])
    BSgate() | (q[8], q[9])
    Interferometer(SWAP) |(q[5], q[9])
    BSgate().H | (q[4], q[5])
    BSgate().H | (q[8], q[9])
    '''
    #PBS3
    BSgate() | (q[2], q[3])
    BSgate() | (q[4], q[5])
    Rgate(np.pi/2) | q[3]
    Rgate(np.pi/2) | q[5]
    Interferometer(PBS) | (q[3], q[5])
    Rgate(-np.pi/2) | q[3]
    Rgate(-np.pi/2) | q[5]
    BSgate().H | (q[2], q[3])
    BSgate().H | (q[4], q[5])
    '''
    '''
    Rgate(np.pi/2) | q[2]
    BSgate() | (q[2], q[3])
    #Interferometer(SWAP) | (q[0], q[1])
    Rgate(-np.pi/2) | q[2]
    Rgate(-np.pi/2) | q[3]
    
    Rgate(np.pi/2) | q[4]
    BSgate() | (q[4], q[5])
    #Interferometer(SWAP) | (q[2], q[3])
    Rgate(-np.pi/2) | q[4]
    Rgate(-np.pi/2) | q[5]
    
    Rgate(np.pi/2) | q[6]
    BSgate() | (q[6], q[7])
    #Interferometer(SWAP) | (q[2], q[3])
    Rgate(-np.pi/2) | q[6]
    Rgate(-np.pi/2) | q[7]
    '''
    BSgate() | (q[2], q[4])
    BSgate() | (q[3], q[5])
    
    BSgate() | (q[4], q[6])
    BSgate() | (q[5], q[7])
    '''
    Rgate(np.pi/2) | q[2]
    Rgate(np.pi/2) | q[3]
    BSgate().H | (q[2], q[3])
    Rgate(-np.pi/2) | q[2]
    
    Rgate(np.pi/2) | q[4]
    Rgate(np.pi/2) | q[5]
    BSgate().H | (q[4], q[5])
    Rgate(-np.pi/2) | q[4]
    
    Rgate(np.pi/2) | q[6]
    Rgate(np.pi/2) | q[7]
    BSgate().H | (q[6], q[7])
    Rgate(-np.pi/2) | q[6]   
    '''
    # X on target
    #Interferometer(SWAP) | (q[8], q[9])
    

ps, pss = AncillaStates(4, 6)
ket = None
for i in range(len(ps)):
    print(ps[i])
    post_select = [ps[i],[0,1,None,None,None,None,None,None,2,3]]
    
    ket = evaluate(Li_mod, target_state, fail_state, post_select=post_select, cutoff_dim=6, title = 'Li et al. 2021 modified - Measure: ' + pss[i], path = 'Li_mod2/Li_mod_'+pss[i], verbosity = 1, return_ket = True, ket = ket)

#%% Adapted Li Version 2

initial_state = [(1,0,1,1,1,0,1,0,1,1,1,0),
                 (1,0,1,1,1,0,1,0,1,1,0,1),
                 (0,1,1,1,1,0,1,0,1,1,1,0),
                 (0,1,1,1,1,0,1,0,1,1,0,1)]

initial_state = FockBasis(initial_state, 4, 12)

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

Li_mod = sf.Program(12)

with Li_mod.context as q:
    Ket(initial_state)  | q
    
    # Preperation of Ancilla states
    BSgate() | (q[2], q[3])
    BSgate() | (q[4], q[5])
    BSgate() | (q[8], q[9])
    
    # PBS1
    Interferometer(SWAP) | (q[1], q[5])
    
    #PBS2
    BSgate() | (q[6], q[7])
    Interferometer(SWAP) | (q[6], q[7])
    BSgate() | (q[10], q[11])
    Interferometer(SWAP) | (q[10], q[11])
    
    Interferometer(SWAP) |(q[7], q[11])
    
    Interferometer(SWAP) | (q[6], q[7])
    BSgate().H | (q[6], q[7])
    Interferometer(SWAP) | (q[10], q[11])
    BSgate().H | (q[10], q[11])
    
    BSgate() | (q[4], q[6])
    BSgate() | (q[5], q[7])
    
    BSgate() | (q[2], q[4])
    BSgate() | (q[3], q[5])
    
    BSgate() | (q[6], q[8])
    BSgate() | (q[7], q[9])

ps, pss = AncillaStates(6, 8)
ket = None

for i in range(len(ps)):
    if 6 in list(ps[i]) or  5 in list(ps[i]) or 4 in list(ps[i]):
        continue
    
    print(ps[i])
    post_select = [list(ps[i]),[0,1,None,None,None,None,None,None,None,None,2,3]]
        
    ket = evaluate(Li_mod, target_state, fail_state, post_select = post_select, ket = ket, cutoff_dim=4, title = 'Li et al. 2021 modified - Measure: ' + pss[i], path = 'Li_mod2/Li_mod2_'+pss[i], verbosity = 1)


#%% Bell State measurement

initial_state =  []

for i in range(4):
    initial_state.append(np.zeros([3]*4, dtype=np.complex64))
    
initial_state[0][1,0,1,0] = 1/np.sqrt(2)
initial_state[0][0,1,0,1] = 1/np.sqrt(2)
initial_state[1][1,0,1,0] = 1/np.sqrt(2)
initial_state[1][0,1,0,1] = - 1/np.sqrt(2)
initial_state[2][1,0,0,1] = 1/np.sqrt(2)
initial_state[2][0,1,1,0] = 1/np.sqrt(2)
initial_state[3][1,0,0,1] = 1/np.sqrt(2)
initial_state[3][0,1,1,0] = - 1/np.sqrt(2)

initial_state = tf.constant(initial_state, dtype = np.complex64)

BSM = sf.Program(4)

with BSM.context as q:
    
    Ket(initial_state)  | q
    
    #PBS3
    BSgate() | (q[0], q[1])
    BSgate() | (q[2], q[3])
    Rgate(np.pi/2) | q[1]
    Rgate(np.pi/2) | q[3]
    Interferometer(PBS) | (q[1], q[3])
    Rgate(-np.pi/2) | q[1]
    Rgate(-np.pi/2) | q[3]
    BSgate().H | (q[0], q[1])
    BSgate().H | (q[2], q[3])
    
evaluate(BSM)

#%% 3/4 BSM

initial_state =  []

for i in range(4):
    initial_state.append(np.zeros([7]*8, dtype=np.complex64))
    
initial_state[0][1,1,1,0,1,0,1,1] = 1/np.sqrt(2)
initial_state[0][1,1,0,1,0,1,1,1] = 1/np.sqrt(2)
initial_state[1][1,1,1,0,1,0,1,1] = 1/np.sqrt(2)
initial_state[1][1,1,0,1,0,1,1,1] = - 1/np.sqrt(2)
initial_state[2][1,1,1,0,0,1,1,1] = 1/np.sqrt(2)
initial_state[2][1,1,0,1,1,0,1,1] = 1/np.sqrt(2)
initial_state[3][1,1,1,0,0,1,1,1] = 1/np.sqrt(2)
initial_state[3][1,1,0,1,1,0,1,1] = - 1/np.sqrt(2)

initial_state = tf.constant(initial_state, dtype = np.complex64)

BSM = sf.Program(8)

with BSM.context as q:
    
    Ket(initial_state)  | q
    
    # Prepare Ancilla
    BSgate() | (q[0], q[1])
    Rgate(np.pi/2) | q[0]
    
    BSgate() | (q[6], q[7])
    Rgate(np.pi/2) | q[6]
    
    BSgate() | (q[2], q[4])
    BSgate() | (q[3], q[5])
    
    BSgate() | (q[4], q[6])
    BSgate() | (q[5], q[7])
    '''
    #PBS3
    BSgate() | (q[0], q[1])
    BSgate() | (q[2], q[3])
    Rgate(np.pi/2) | q[1]
    Rgate(np.pi/2) | q[3]
    Interferometer(PBS) | (q[1], q[3])
    Rgate(-np.pi/2) | q[1]
    Rgate(-np.pi/2) | q[3]
    BSgate().H | (q[0], q[1])
    BSgate().H | (q[2], q[3])
    
    #PBS4
    BSgate() | (q[2], q[3])
    BSgate() | (q[4], q[5])
    Rgate(np.pi/2) | q[3]
    Rgate(np.pi/2) | q[5]
    Interferometer(PBS) | (q[3], q[5])
    Rgate(-np.pi/2) | q[3]
    Rgate(-np.pi/2) | q[5]
    BSgate().H | (q[2], q[3])
    BSgate().H | (q[4], q[5])
    '''
evaluate(BSM, cutoff_dim = 7)

#%% Li et al. 2021

initial_state = [(1,0,1,0,1,0,1,0,1,1),
                 (1,0,1,0,1,0,0,1,1,1),
                 (0,1,1,0,1,0,1,0,1,1),
                 (0,1,1,0,1,0,0,1,1,1)]

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

post_select = [[slice(None),slice(None),slice(None),slice(None),slice(None),slice(None)],[0,1,None,None,None,None,2,3,None,None]]

@operation(10)
def Li_et_al_2021(q):   
    # Preperation of Ancilla states
    BSgate() | (q[2], q[3])
    
    # PBS1
    Interferometer(SWAP) | (q[1], q[3])
    
    #PBS2
    BSgate() | (q[4], q[5])
    BSgate() | (q[6], q[7])
    Interferometer(SWAP) |(q[5], q[7])
    BSgate().H | (q[4], q[5])
    BSgate().H | (q[6], q[7])
    
    #PBS3
    BSgate() | (q[2], q[3])
    BSgate() | (q[4], q[5])
    Rgate(np.pi/2) | q[3]
    Rgate(np.pi/2) | q[5]
    Interferometer(SWAP) | (q[3], q[5])
    Rgate(-np.pi/2) | q[3]
    Rgate(-np.pi/2) | q[5]
    BSgate().H | (q[2], q[3])
    BSgate().H | (q[4], q[5])
    
qgs = QGS(6, initial_state, layers=1, decomposition='Reck', modes=10, cutoff_dim = 5)
qgs.fit(target_state, fail_states = fail_state, post_select = post_select, preparation = Li_et_al_2021, reps = 1000, path = 'Li_opt.npz')

#%% 4 photons 8 modes

initial_state = [(1,0,1,0,1,0,1,0),
                 (1,0,1,0,1,0,0,1),
                 (0,1,1,0,1,0,1,0),
                 (0,1,1,0,1,0,0,1)]

# output state:
target_state =  [(1,0,1,0),
                 (1,0,0,1),
                 (0,1,0,1),
                 (0,1,1,0)]

# failed states:
fail_state = [(1,1,0,0),
              (0,0,1,1),
              (2,0,0,0),
              (0,2,0,0),
              (0,0,2,0),
              (0,0,0,2)]

post_select = [[slice(None),slice(None),slice(None),slice(None)],[0,1,None,None,None,None,2,3]]

@operation(8)
def Li_et_al_2021(q):
    # Preperation of Ancilla states
    BSgate() | (q[2], q[3])

    # PBS1
    Interferometer(SWAP) | (q[1], q[3])
    
    #PBS2
    BSgate() | (q[4], q[5])
    BSgate() | (q[6], q[7])
    Interferometer(SWAP) |(q[5], q[7])
    BSgate().H | (q[4], q[5])
    BSgate().H | (q[6], q[7])
    
qgs = QGS(4, initial_state, layers=1, decomposition='Reck', modes=8)
qgs.fit(target_state, fail_states = fail_state, post_select = post_select, preparation = Li_et_al_2021, reps = 500, n_sweeps=56, path = '4p8m', sweep_low = 0.008, sweep_high = 0.25)

#%% Quantum Non-Demulition Measurement (QND):

initial_state = [(1,1,0,0),
                 (1,0,1,0),
                 (1,0,0,1),
                 (0,1,1,0),
                 (0,1,0,1),
                 (0,0,1,1)]

initial_state = FockBasis(initial_state, 3, 4)

QND = sf.Program(4)

with QND.context as q:
    Ket(initial_state)  | q
    #BSgate() | (q[2], q[3])
    # PBS1 H/V
    # Transmitts H state
    # Refelcts V state
    '''
    Interferometer(SWAP) | (q[0], q[2])
    
    '''
    #PBS2 +/-
    '''
    # Conversion from H/V to +/- Basis
    BSgate() | (q[0], q[1])
    #Interferometer(SWAP) | (q[0], q[1])
    BSgate() | (q[2], q[3])
    #Interferometer(SWAP) | (q[2], q[3])
    # PBS
    Interferometer(SWAP) | (q[0], q[2])
    # Conversion from +/- to H/V Basis
    #Interferometer(SWAP) | (q[0], q[1])
    BSgate().H | (q[0], q[1])
    #Interferometer(SWAP) | (q[2], q[3])
    BSgate().H | (q[2], q[3])
    
    '''
    #PBS3 R/L
    # Conversion from H/V to R/L basis
    Rgate(np.pi/2) | q[0]
    BSgate() | (q[0], q[1])
    #Interferometer(SWAP) | (q[0], q[1])
    Rgate(-np.pi/2) | q[0]
    Rgate(-np.pi/2) | q[1]
    
    Rgate(np.pi/2) | q[2]
    BSgate() | (q[2], q[3])
    #Interferometer(SWAP) | (q[2], q[3])
    Rgate(-np.pi/2) | q[0]
    Rgate(-np.pi/2) | q[1]
    
    
    # PBS
    #Interferometer(SWAP) | (q[1], q[3])
    # Conversion from R/L to H/V basis
    #Rgate(-np.pi/2) | q[1]
    #Interferometer(SWAP) | (q[0], q[1])
    #BSgate().H | (q[0], q[1])
    #Rgate(-np.pi/2) | q[3]
    #Interferometer(SWAP) | (q[2], q[3])
    #BSgate().H | (q[2], q[3])
    
    '''
    BSgate() | (q[1], q[2])
    
    BSgate() | (q[0], q[1])
    BSgate() | (q[2], q[3])
    
    BSgate() | (q[1], q[2])
    
    BSgate() | (q[0], q[3])
    '''
    
#post_select = [[1,1,0],[0,None,None,None]]
    
_ = evaluate(QND, gate_cutoff = 6, cutoff_dim = 3)

#%% Fail state correction:
    
initial_state = [(1,0,1,1,1,1,1),
                 (0,1,1,1,1,1,1),
                 (1,1,1,1,1,1,1)]

target_state = [(1,0,1),
                (0,1,1),
                (slice(None),slice(None),0)]

qgs = QGS(7, initial_state, gate_cutoff = 3, modes = 7)
qgs.fit(target_state, reps=1000, path = 'FSC_5p5m/model.npz')

ps, pss = AncillaStates(4, 4, False)
ket = None
for i in range(len(ps)):
    print(pss[i])
    post_select = [ps[i],[0,1,2,None,None,None,None]]
    ket = qgs.evaluate(ket, target_state, permutation=[0,1,2], path = 'FSC_5p5m/' + pss[i], title = 'Fail state correction - Measure: ' + pss[i], post_select = post_select, verbosity = 1)
    
#%% Fail state correction:
    
initial_state = [(1,0,1,1,1,1),
                 (0,1,1,1,1,1),
                 (1,1,1,1,1,1)]

target_state = [(1,0),
                (0,1),
                (0,0)]

qgs = QGS(6, initial_state, gate_cutoff = 3, modes = 6)
qgs.fit(target_state, reps=1000, path = 'FSC_4p4m/model.npz')

ps, pss = AncillaStates(4, 4, False)
ket = None
for i in range(len(ps)):
    print(pss[i])
    post_select = [ps[i],[0,1,None,None,None,None]]
    ket = qgs.evaluate(ket, target_state, permutation=[0,1,2], path = 'FSC_4p4m/' + pss[i], title = 'Fail state correction - Measure: ' + pss[i], post_select = post_select, verbosity = 1)
    
#%% QND:
    
initial_state = [(0,0,1,1),
                 (1,0,1,1),
                 (2,0,1,1)]

target_state = [(0,slice(None)),
                (1,0),
                (0,slice(None))]

qgs = QGS(4, initial_state, gate_cutoff = 3, modes = 4)
qgs.fit(target_state, reps=1000, path = 'QND_2p4m/model.npz')

ps, pss = AncillaStates(4, 2, False)
ket = None
for i in range(len(ps)):
    print(pss[i])
    post_select = [ps[i],[0,1,None,None]]
    ket = qgs.evaluate(ket, target_state, permutation=[0,1,2], path = 'QND_2p4m/' + pss[i], title = 'QND - Measure: ' + pss[i], post_select = post_select, verbosity = 1)
    