#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:37:31 2023

@author: oqi
"""
from QGS import *

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import Ket, BSgate, Rgate, Interferometer, MZgate
from strawberryfields.utils import operation

# Permutation matrix
PBS = np.array([[0, 1], [1, 0]])

#%% 6 Photons on 9 modes - sufficient for EPR generation
# input state:
initial_state = [(1,0,1,0,1,1,1,0,1),
                 (1,0,0,1,1,1,1,0,1),
                 (0,1,1,0,1,1,1,0,1),
                 (0,1,0,1,1,1,1,0,1)]

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
        Interferometer(PBS) | (q[1], q[3])
        
        #PBS2
        BSgate() | (q[4], q[5])
        BSgate() | (q[6], q[7])
        Interferometer(PBS) |(q[5], q[7])
        BSgate().H | (q[4], q[5])
        BSgate().H | (q[6], q[7])
        
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
        
    evaluate(Li2021, target_state, fail_state, post_select=post_select,cutoff_dim=4, title = 'Li et al. 2021 - Measure: ' + pss[i], path = 'Li2021/Li2021_'+pss[i])
#%% Adapted Li

initial_state = [(1,0,1,0,1,0,1,1,1,0),
                 (1,0,1,0,1,0,1,1,0,1),
                 (0,1,1,0,1,0,1,1,1,0),
                 (0,1,1,0,1,0,1,1,0,1)]

initial_state = FockBasis(initial_state, 5, 10)

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

ps, pss = AncillaStates(4, 6)

for i in range(len(ps)):
    print(ps[i])
    post_select = [ps[i],[0,1,None,None,None,None,None,None,2,3]]
    
    Li_mod = sf.Program(10)
    
    with Li_mod.context as q:
        Ket(initial_state)  | q
        
        # Preperation of Ancilla states
        BSgate() | (q[2], q[3])
        
        # PBS1
        Interferometer(PBS) | (q[1], q[3])
        
        #PBS2
        BSgate() | (q[4], q[5])
        BSgate() | (q[8], q[9])
        Interferometer(PBS) |(q[5], q[9])
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
        BSgate() | (q[2], q[4])
        BSgate() | (q[3], q[5])
        
        BSgate() | (q[4], q[6])
        BSgate() | (q[5], q[7])
        
        # X on target
        Interferometer(PBS) | (q[8], q[9])
        
    evaluate(Li_mod, target_state, fail_state, post_select=post_select, cutoff_dim=5, title = 'Li et al. 2021 modified - Measure: ' + pss[i], path = 'Li_mod/Li_mod_'+pss[i])


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
    initial_state.append(np.zeros([5]*6, dtype=np.complex64))
    
initial_state[0][1,0,1,0,1,1] = 1/np.sqrt(2)
initial_state[0][0,1,0,1,1,1] = 1/np.sqrt(2)
initial_state[1][1,0,1,0,1,1] = 1/np.sqrt(2)
initial_state[1][0,1,0,1,1,1] = - 1/np.sqrt(2)
initial_state[2][1,0,0,1,1,1] = 1/np.sqrt(2)
initial_state[2][0,1,1,0,1,1] = 1/np.sqrt(2)
initial_state[3][1,0,0,1,1,1] = 1/np.sqrt(2)
initial_state[3][0,1,1,0,1,1] = - 1/np.sqrt(2)

initial_state = tf.constant(initial_state, dtype = np.complex64)

BSM = sf.Program(6)

with BSM.context as q:
    
    Ket(initial_state)  | q
    
    # Prepare Ancilla
    BSgate() | (q[4], q[5])
    Rgate(np.pi/2) | q[4]
    '''
    BSgate() | (q[0], q[2])
    BSgate() | (q[1], q[3])
    
    BSgate() | (q[2], q[4])
    BSgate() | (q[3], q[5])
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
    
evaluate(BSM, cutoff_dim = 5)

#%% Li et al. 2021

initial_state = [(1,0,1,0,1,0,1,0,1,1),
                 (1,0,1,0,1,0,0,1,1,1),
                 (0,1,1,0,1,0,1,0,1,1),
                 (0,1,1,0,1,0,0,1,1,1)]

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

post_select = [[slice(None),slice(None),slice(None),slice(None), slice(None),slice(None)],[0,1,None,None,None,None,2,3,None,None]]

Li2021 = sf.Program(10)

with Li2021.context as q:
    Ket(initial_state)  | q
    '''
    # Preperation of Ancilla states
    BSgate() | (q[2], q[3])
    
    # PBS1
    Interferometer(PBS) | (q[1], q[3])
    
    #PBS2
    BSgate() | (q[4], q[5])
    BSgate() | (q[6], q[7])
    Interferometer(PBS) |(q[5], q[7])
    BSgate().H | (q[4], q[5])
    BSgate().H | (q[6], q[7])
    
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
    # Preperation of Ancilla states
    BSgate() | (q[8], q[9])
    

evaluate(Li2021, target_state, fail_state, post_select=post_select,cutoff_dim=4, title = 'Li et al. 2021', path = 'Li2021')
#%%
@operation(8)
def Li_et_al_2021(q):
    # Preperation of Ancilla states
    #BSgate() | (q[2], q[3])
    
    # PBS1
    #Interferometer(PBS) | (q[1], q[3])
    
    #PBS2
    BSgate() | (q[4], q[5])
    BSgate() | (q[6], q[7])
    