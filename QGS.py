# -*- coding: utf-8 -*-
"""
Quantum Gate Synthesis Module

Version: 2023-01-20

@author: Martin F.X. Mauser
"""

import os

import numpy as np

import strawberryfields as sf
from strawberryfields import ops


import tensorflow as tf
from tensorflow.python.framework.ops import Tensor, EagerTensor
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from matplotlib import pyplot as plt
import tqdm
import itertools
from contextlib import redirect_stdout

from typing import Optional, Union, Callable

#
#TODO Memory allocation optimization (eg. Tensorflow Large-Modell Support (TFLMS) from IBM)
#

#%% Hepler - Functions:
    
def FockBasis(state_list: list, cutoff_dim: int, modes: int, dtype: type = np.complex64, multiplex: bool = False) -> np.ndarray: 
    """
    Create numpy.ndarray out of given indizes representing a specific state in Fock space
    
    Parameters
    ----------
    state_list : list
        list of tuple containing a states in the truncated Fock Space (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
    cutoff_dim : int  
        truncation of the Fock Space
    modes : int       
        number of modes in the truncated Fock Space
    dtype : type, optional     
        data type of the returned state (e.g. np.complex64 for standard states, bool for masks). The default is np.complex64
    multiplex: bool, optional
        Defines if multiple post selection states should be considered. The default is False.
       
    Returns
    -------
    numpy.ndarray
        complete numpy array to be handeld by Strawberryfields
        
    """
    state = []
    if multiplex:
        for _ in range(len(state_list[0])):
            state.append(np.zeros([cutoff_dim]*modes, dtype=dtype))
        
        for p in range(len(state_list)):
            for i in range(len(state_list[p])):
                state[i][state_list[p][i]] = 1
    else:
        for _ in range(len(state_list)):
            state.append(np.zeros([cutoff_dim]*modes, dtype=dtype))
        
        for i in range(len(state_list)):
            state[i][state_list[i]] = 1
    
    return np.array(state)

def PostSelect(state_list: list, ps: list, mask: Optional[list] = None, multiplex: bool = False) -> list:
    """
    Appends (specific) ancilla states to every state in the provided state list, according to the optinal mask.

    Parameters
    ----------
    state_list : list
        list of tuple containing a states in the truncated Fock Space (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
    ps : list
        ancilla states to append to the states in state_list (this can either be a list of well defined states or arbitary ones - [0,1], [0, slice(None)], ...)
    mask : Optional[list], optional
        Mask defining the position of each individual computaion qubit (integer) and the position of the ancilla qubits (None). The default is None, meaning no mask provided.
    multiplex: bool, optional
        Defines if multiple post selection states should be considered. The default is False.
        
    Returns
    -------
    list
        New list of states, where each state now also contains information about the ancilla qubits

    """
    state_l = state_list.copy()
    
    if multiplex:
        
        state_matrix = []
        for p in range(len(ps)):
            state_matrix.append([])
            for i in range(len(state_list)):
                state_matrix[p].append([])
                if mask is None:
                    state_matrix[p][i] = list(state_list[i]) + ps[p]
                else:
                    temp = mask.copy()
                    k = 0
                    for j in range(len(temp)):
                        if temp[j] is None: # Masking the postion of an ancilla qubit
                            temp[j] = ps[p][k]
                            k += 1
                        else:
                            temp[j] = state_list[i][int(temp[j])] # Assigning the value of the labeld qubit
                    
                    state_matrix[p][i] = temp
                    
                state_matrix[p][i] = tuple(state_matrix[p][i])
                
        return state_matrix
    
    else:
        for i in range(len(state_list)):
            if mask is None:
                state_l[i] = list(state_list[i]) + ps
            else:
                temp = mask.copy()
                k = 0
                for j in range(len(temp)):
                    if temp[j] is None: # Masking the postion of an ancilla qubit
                        temp[j] = ps[k]
                        k += 1
                    else:
                        temp[j] = state_list[i][int(temp[j])] # Assigning the value of the labeld qubit
                
                state_l[i] = temp
                
            state_l[i] = tuple(state_l[i])
        
        return state_l
 
def AncillaStates(photons: int, modes: int, well_defined: bool = True) -> (list, list):
    """
    Define all possible Ancilla states of the systhem

    Parameters
    ----------
    photons : int
        Number of ancilla photons.
    modes : int
        Number of ancilla modes.
    well_defined : bool
        Only provide ancilla states, where every postion of the photon is well defined.
        The default is true.

    Returns
    -------
    (list, list)
        Returns list of all possible ancilla states once as list of tuples ready to use once as list of strings for easy naming.

    """
    ps = []
    pss = []
    
    if not well_defined:
        perm = itertools.product(list(range(photons)), repeat = modes)
        for a in perm:
            temp = list(a)
            if sum(temp) < photons:
                temp_string = ''.join(str(e) for e in temp)
                for j in range(len(temp)):
                    if temp[j] == 0:
                        temp[j] = slice(None)
                ps.append(tuple(temp))
                pss.append(temp_string.replace('0', 'x'))
                
    perm = itertools.product(list(range(photons + 1)), repeat = modes)
    for a in perm:
        if sum(list(a)) == photons:
            ps.append(a)
            pss.append(''.join(str(e) for e in list(a)))
            
    return ps, pss

def StateMasks(modes: int, keep_order: bool = True, dual_rail: bool = True, comp_modes: int = 4) -> list:
    """
    Gernate all possible Mask for the given number of modes.

    Parameters
    ----------
    modes : int
        Number of modes of the system.
    keep_order : bool, optional
        Keep order of computational modes. The default is True.
    dual_rail : bool, optional
        Keep dual-rail encoding of computational modes (not seperating the modes with ancilla modes). The default is True.
    comp_modes : int, optional
        Number of computational modes. The default is 4.

    Raises
    ------
    Exception
        Number of computational modes must be smaller than the number of total modes.

    Returns
    -------
    perm : list
        List of all possible masks.

    """
    
    mask = [None] * modes
    if modes >= comp_modes:
        for i in range(comp_modes):
            mask[i] = i
    else:
        raise Exception('Number of computational Modes must be higher than the number modes')
            
    perm = list(itertools.permutations(mask))
    
    if keep_order:
        new_perm = []
        
        for p in perm:
            temp = [i for i in list(p) if i is not None]
            if all(a < b for a, b in zip(temp, temp[1:])):
                new_perm.append(p)
                
        perm = new_perm
        
    if dual_rail:
        new_perm = []
        
        for p in perm:
            flag = True
            for i in range(len(p) - 1):
                if p[i] is not None and p[i]%2 == 0:
                    if p[i + 1] != p[i] + 1:
                        flag = False
                        
            if flag:
                new_perm.append(p)
        perm = new_perm
        
    return perm

def SimpleState(state: tuple) -> list:
    """
    Produces simpler more easily readable output state by replacing slice(None) with ':'
    
    Parameters
    ----------
    state : tuple
        state to be simplified
        
    Returns
    -------
        simplified state to print
    """
    return [':' if x == slice(None) else x for x in state]

def report(path: str, *args, **kwargs):
    """
    Either prints to console or to a report file.

    Parameters
    ----------
    path : str
        Path to report file. If set to None it will output in the console.

    """
    if path is None:
        print(*args, **kwargs)
    else:
        with open(path + '.txt', 'a') as f:
            with redirect_stdout(f):
                print(*args, **kwargs)   
                
def evaluate(circuit: sf.Program, target_state: Optional[list] = None, fail_states: Optional[list] = None, post_select: Optional[list] = None, ket: Optional[np.ndarray] = None, backend: str = 'tf', cutoff_dim: int = 3, gate_cutoff: int = 4, path: Optional[str] = None, precision: int = 3, title: Optional[str] = None, permutation: list = [0,1,3,2], verbosity: int = 2, return_ket: bool = False) -> np.ndarray:
    """
    Evaluate given optical cicuit.

    Parameters
    ----------
    circuit : sf.Program
        Optical cicuit for Strawberryfields already containg an input state.
    target_state : Optional[list], optional
        List of tuple containing the target states in the truncated Fock Space in order of the given input states (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
        The default is None.
    fail_states : Optional[list], optional
        List of tuple containing the fail states in the truncated Fock Space (eg. [(1,1,0,0),(2,0,0,0),...] == [|1100>, |2000>, ...]).
        Hereby is a fail state defined as a state in computational modes which do not represent a successful gate opertaion - keyword: photon bunching
        The default is None.
    post_select : Optional[list], optional
            More detailed definition of the ancilla modes. The default is None.
    ket : Optional[np.ndarray], optional
        Provid already calulated ket state for evaluation and skip so the calulation
        The default is None.
    backend : str, optional
        Backend used for the calcuations on the optical system. The default is 'tf'.
    cutoff_dim : int, optional
        Defines the truncation of the Fock space. The default is 3.
    gate_cutoff : int, optional
        Dimensionality of the gate, which should be synthesised. The default is 4.
    path : Optional[str], optional
        Path for saving created plots and reports. The default is None.
    precision : int, optional
        Number of digits after the decimal point to print. The default is 3.
    title : Optional[str], optional
        Titel for the created plots. The default is None.
    permutation : list, optional
        Permutation between input and target states. The default is [0,1,3,2] = CNOT.
    verbosity : int, optional
        Set verbosity for amount of information in the report:
        verbosity = 2 : All information in the report
                    1 : Only post selected states and target state propapility
                    0 : No data reported only visualization
        The default is 2.
    return_ket : bool, optional
        Define if ket should be returned or deleted.
        The default is False.

    Returns
    -------
    ket : np.nd array
        The reulting ket state
    Plots and evaluation report

    """
    if type(ket) != np.ndarray:
        eng = sf.Engine(backend, backend_options={"cutoff_dim": cutoff_dim, "batch_size": gate_cutoff})
        result = eng.run(circuit)
        
        state = result.state
        
        ket = state.ket()
    
    if tf.is_tensor(ket):
        ket = ket.numpy()
        
    if target_state is not None:
        if post_select is not None:
            if len(post_select) == 3:
                target_state = PostSelect(target_state, post_select[0], post_select[1], post_select[2])
                t_mask = FockBasis(target_state, cutoff_dim, circuit.num_subsystems, dtype = bool, multiplex = post_select[2])
            else:
                target_state = PostSelect(target_state, post_select[0], post_select[1])
                t_mask = FockBasis(target_state, cutoff_dim, circuit.num_subsystems, dtype = bool)
        else:
            t_mask = FockBasis(target_state, cutoff_dim, circuit.num_subsystems, dtype = bool)
    
    if fail_states is not None:
        if post_select is not None:
            if len(post_select) == 3:
                fail_states = PostSelect(fail_states, post_select[0], post_select[1], post_select[2])
                fail_mask = FockBasis(fail_states, cutoff_dim, circuit.num_subsystems, dtype = bool, multiplex = post_select[2])
            else:
                fail_states = PostSelect(fail_states, post_select[0], post_select[1])
                fail_mask = FockBasis(fail_states, cutoff_dim, circuit.num_subsystems, dtype = bool)
        else:
            fail_mask = FockBasis(fail_states, cutoff_dim, circuit.num_subsystems, dtype = bool)
        
    # Check the normalization of the ket.
    # This does give the exact answer because of the cutoff we chose.
    for i in range(ket.shape[0]):
        report(path, f"\n{i+1}. Input:\n")        
        
        report(path, "The norm of the ket is ", np.linalg.norm(ket[i]))
        
        ket_rounded = np.round(ket[i], precision)
        ind = np.array(np.nonzero(np.real_if_close(ket_rounded))).T
        # And these are their coefficients
        
        if verbosity >= 2:
            report(path, "\nThe nonzero components are:")
            for state in ind:
                report(path, ket_rounded[tuple(state)], tuple(state))
            
        if post_select is not None and verbosity >= 1:
            if len(post_select) == 3 and post_select[2]:
                for p in post_select[0]:
                    ps = PostSelect([[slice(None)]*(circuit.num_subsystems - len(p))], p, post_select[1])
                    
                    ps_mask = FockBasis(ps, cutoff_dim, circuit.num_subsystems, dtype = bool)
                    #ps_ket = np.round(tf.boolean_mask(ket, ps_mask[0], axis = 1)[i], precision)
                    #ps_ket = np.round(tf.where(ps_mask[0], ket[i], tf.zeros_like(ket[i])), precision)
                    ps_ket = np.round(np.where(ps_mask[0], ket[i], np.zeros_like(ket[i])), precision)
                    p_post_select = np.round(np.linalg.norm(ps_ket) ** 2, precision*2)  # Check the probability of this event
                    del ps_mask
                    
                    report(path, "\nPost selecting: ", SimpleState(tuple(ps[0])))
                    report(path, "\nThe probability for post selecting is ", p_post_select)
                    # These are the only nonzero components
                    ind = np.array(np.nonzero(np.real_if_close(ps_ket))).T
                    report(path, "\nPost selected states:")
                    for state in ind:
                        report(path, ps_ket[tuple(state)], SimpleState(tuple(state)))
                
            else:
                ps = PostSelect([[slice(None)]*(circuit.num_subsystems - len(post_select[0]))], post_select[0], post_select[1])
                
                ps_mask = FockBasis(ps, cutoff_dim, circuit.num_subsystems, dtype = bool)
                #ps_ket = np.round(tf.boolean_mask(ket, ps_mask[0], axis = 1)[i], precision)
                #ps_ket = np.round(tf.where(ps_mask[0], ket[i], tf.zeros_like(ket[i])), precision)
                ps_ket = np.round(np.where(ps_mask[0], ket[i], np.zeros_like(ket[i])), precision)
                p_post_select = np.round(np.linalg.norm(ps_ket) ** 2, precision*2)  # Check the probability of this event
                del ps_mask
                
                report(path, "\nPost selecting: ", SimpleState(tuple(ps[0])))
                report(path, "\nThe probability for post selecting is ", p_post_select)
                # These are the only nonzero components
                ind = np.array(np.nonzero(np.real_if_close(ps_ket))).T
                report(path, "\nPost selected states:")
                for state in ind:
                    report(path, ps_ket[tuple(state)], SimpleState(tuple(state)))
            
        if target_state is not None and verbosity >= 1:
            if post_select is not None and len(post_select) == 3 and post_select[2]:
                target_ket = np.round(tf.boolean_mask(ket, t_mask[i], axis = 1)[i], precision)
                for p in range(len(post_select[0])):
                    report(path, "\nTarget state:",SimpleState(target_state[p][i]))
                    
                p_target_state = np.round(np.linalg.norm(target_ket) ** 2, precision*2)
                report(path,"The overall propability to get the target state is", p_target_state)
            
            else:
                target_ket = np.round(tf.boolean_mask(ket, t_mask[i], axis = 1)[i], precision)
                report(path, "\nTarget state:",SimpleState(target_state[i]))
                p_target_state = np.round(np.linalg.norm(target_ket) ** 2, precision*2)
                report(path,"The overall propability to get the target state is", p_target_state)
            
        if verbosity >= 1:
            report(path,"\n------------------------------------------------------------------------\n")
        
    if target_state is not None:   
                
        A = tf.stack([[tf.math.reduce_sum(tf.abs(tf.boolean_mask(ket, t_mask[i], axis = 1))[j]) for j in range(gate_cutoff)] for i in range(gate_cutoff)]).numpy()[:,permutation]
        
        if fail_states is not None:
            B = tf.stack([[tf.math.reduce_sum(tf.abs(tf.boolean_mask(ket, fail_mask[i], axis = 1))[j]) for j in range(gate_cutoff)] for i in range(fail_mask.shape[0])]).numpy()
        else:
            B = None
             
        fig2 = plt.figure(figsize=(10,5))
        if title is not None:
            fig2.suptitle(title)
            
        if B is None:
            ax1 = fig2.add_subplot(111, projection='3d')
        else:
            ax1 = fig2.add_subplot(121, projection='3d')
            ax2 = fig2.add_subplot(122, projection='3d')
        
        if gate_cutoff == 4:
            #
            #TODO Add invidualization for other gate_cutoffs, e.g. with list for labels and colors
            #
            # For classic CNOT operation
            state_label = ['|HH>','|HV>','|VH>','|VV>']
            f_state_label = ['|1100>', '|0011>', '|2000>', '|0200>', '|0020>', '|0002>']
            
            cs = [[c]*4 for c in ['r','g','b','c']]
            cs = list(np.array(cs).flatten())
        
        xpos = np.arange(0,gate_cutoff,1)
        ypos = np.arange(0,gate_cutoff,1)
        xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)
        
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(gate_cutoff * gate_cutoff)
        
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = np.absolute(A.flatten())
        
        ax1.set_xticks([x + 0.5 for x in list(range(gate_cutoff))])
        ax1.set_yticks([x + 0.5 for x in list(range(gate_cutoff))])  
        
        if gate_cutoff == 4:
            ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)
            ax1.w_xaxis.set_ticklabels(state_label)
            ax1.w_yaxis.set_ticklabels(state_label)
        else:
            ax1.bar3d(xpos, ypos, zpos, dx, dy, dz)
            
        ax1.set_xlabel('output states')
        ax1.set_ylabel('input states')
        ax1.set_zlabel('sqrt(amplitude)')
        ax1.set_zlim([0.0,1.0])
        
        if B is not None:
            xpos = np.arange(0,6,1)
            ypos = np.arange(0,4,1)
            xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)
            
            xpos = xpos.flatten()
            ypos = ypos.flatten()
            zpos = np.zeros(6*4)
            
            dx = 0.5 * np.ones_like(zpos)
            dy = dx.copy()
            dz = np.absolute(B.flatten())
            
            cs = [[c]*6 for c in ['r','g','b','c']]
            cs = list(np.array(cs).flatten())
            
            ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)
            ax2.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5])
            ax2.w_xaxis.set_ticklabels(f_state_label)
            ax2.set_yticks([0.5,1.5,2.5,3.5])
            ax2.w_yaxis.set_ticklabels(state_label)
            ax2.set_xlabel('output states')
            ax2.set_ylabel('input states')
            ax2.set_zlabel('sqrt(amplitude)')
            ax2.set_zlim([0.0,1.0])
        
        if path is not None:
            fig2.savefig(path)
        else:
            fig2.show()
            
        plt.close(fig2)
            
    if return_ket:
        return ket
    else:
        del ket
        
#TODO 
        
def Conv_Test(photons, init_state, target, optimizer, modes = None, decomposition='Clements', post_select = None, input_state = None, gate_cutoff = 4, cutoff_dim = None, layers = 1, repeat = 10, steps = 500, p_success = 1, fail_state = None, punish = 0.1, conv_crit = 50, delta_step = 0.001, path = None, ancilla = False, fail = False):
    
    if path is not None and not os.path.exists(path):
        os.mkdir(path)
        
    steps = []
    loss_list = []
    loss = []
    best_loss = np.inf
        
    with tqdm.tqdm(range(1,repeat+1)) as pbar:        
        for i in pbar:
            qgs = QGS(photons, init_state, modes = modes, post_select = post_select, input_state = input_state, gate_cutoff = gate_cutoff, cutoff_dim = cutoff_dim, layers = layers, decomposition = decomposition)
            qgs.fit(target, post_select=post_select, steps=steps, fail_state = fail_state, optimizer = optimizer, silence = True, p_success = p_success, punish = punish, conv_crit = conv_crit, delta_step = delta_step, ancilla = ancilla, fail = fail)
            steps.append(len(qgs.cost_progress))
            loss_list.append(qgs.cost_progress)
            loss.append(qgs.cost_progress[-1])
            
            if loss[-1] < best_loss:
                best_loss = loss[-1]
            
                if path is not None:
                    qgs.save(path+'\\best_model')
                    
            pbar.set_postfix({'Best loss':float(best_loss)})
            
    fig, axs = plt.subplots(1,2,figsize=(12, 6))
    fig.suptitle(str(optimizer.get_config()))
    for l in loss_list:  
        axs[0].plot(l,'b-',alpha=0.125)
    axs[0].set_ylabel('Cost')
    axs[0].set_xlabel('Step')
    
    steps_mean = np.mean(steps)
    steps_std = np.std(steps)
    
    axs[1].axvline(np.min(steps), color = 'red', alpha = 0.5, lw = 0.5)
    axs[1].axvline(np.max(steps), color = 'red', alpha = 0.5, lw = 0.5)
    axs[1].axvline(steps_mean, color = 'orange', lw = 1)
    axs[1].axvspan(steps_mean-steps_std, steps_mean+steps_std, color = 'orange', alpha = 0.1)
    
    loss_mean = np.mean(loss)
    loss_std = np.std(loss)
    
    axs[1].axhline(np.min(loss), color = 'red', alpha = 0.5, lw = 0.5)
    axs[1].axhline(np.max(loss), color = 'red', alpha = 0.5, lw = 0.5)
    axs[1].axhline(loss_mean, color = 'orange', lw = 1)
    axs[1].axhspan(loss_mean-loss_std, loss_mean+loss_std, color = 'orange', alpha = 0.1)
    
    axs[1].scatter(steps,loss, c='b', alpha=0.25)
    axs[1].set_ylabel('Cost')
    axs[1].set_xlabel('Step')
    
    if path is not None:
        plt.savefig(path+'\\opt_perform')
    else:
        plt.show()
        
    plt.close(fig)
            
#%% QGS - Class:

class QGS:
    def __init__(self, photons: int, init_state: Union[list, np.ndarray, Tensor, EagerTensor], modes: Optional[int] = None, gate_cutoff: int = 4, cutoff_dim: Optional[int] = None, layers: int = 1, decomposition: str = 'Clements') -> None:
        """
        Initializes Quantum Gate Synthesis Class

        Parameters
        ----------
        photons : int
            Total number of Photons in the sythem
        init_state : Union[list, np.ndarray, Tensor, EagerTensor]
            input state for the systhem either provided as a
                ... list of tuple containing a states in the truncated Fock Space (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
                ... numpy.ndarray in the correct format to be hangled by Strawberryfields as a state in a trancated FockSpace
                ... Tensor in the correct format to be hangled by Strawberryfields as a state in a trancated FockSpace
        modes : Optional[int], optional
            Number of modes in the truncated Fock Space. The default is given as a dual-rail-encoding of the total number of photons.
        gate_cutoff : int, optional
            Dimensionality of the gate, which should be synthesised. The default is 4.
        cutoff_dim : Optional[int], optional
            Defines the truncation of the Fock space. The default is the total number of photons + 1, toensure calculation consitancy.
        layers : int, optional
            Number of layers used to synthesie the gate. The default is 1.
        decomposition : str, optional
            Decompostion used in each layer to create a unitary operation on the photonic Systhem. The default is 'Clements'.

        Raises
        ------
        Exception
            Conflicts in provided variables.

        Returns
        -------
        None.

        """
        
        # number of photons:
        self.photons = photons
        
        # number of modes:
        # defaults to dual rail encoding
        if modes is None:
            self.modes = photons * 2
        else:
            self.modes = modes
            
        # cutoff dimension:
        # defines the truncation in fock space
        # should be set to (1 + total number of photons) to ensure calulation consistency
        if cutoff_dim is None:
            self.cutoff_dim = int(photons + 1)
        else:
            self.cutoff_dim = cutoff_dim
            
        # gate cutoff:
        # defines the dimension of the underlying gate, which should be created
        self.gate_cutoff = gate_cutoff
        
        # depth of a single layer:
        # describes the number of tuneable parameters
        self.depth = int(self.modes * (self.modes - 1)//2)
        
        # number of layers:
        self.layers = layers
        
        # structure (decomposition) of a general unitary gate for photons
        # repeted structure of MZ Interferometer for universal gate
        # Clements et al. (2016) [2] - more balanced than Reck et al. (1994)[3]
        self.decomposition = decomposition
        self.struct = []
        if self.decomposition == 'Clements':
            for _ in range(self.modes//2):
                for i in range(2):
                    for j in range(self.modes-1):
                        if j%2==i:
                            self.struct.append(j)
                            
            if self.modes%2 == 1:
                for j in range(self.modes-1):
                    if j%2==0:
                        self.struct.append(j)
                        
        # Initialisation:
            
        # Machine Learning preparation flag:
        self.ML_initialised = False
        
        # Input state:
        if type(init_state) == list:
            if len(init_state) != self.gate_cutoff:
                raise Exception("Number of input states must match the defined gate_cutoff")
            self.init_state = FockBasis(init_state, self.cutoff_dim, self.modes)
        else:
            self.init_state = init_state
        
        # Target state:
        self.target_state = None  
        
        # Fail states:
        # States in computational modes, which do not represent a successsful gate operation
        self.fail_state = None
        
    def init_weights(self, passive_sd: float = 0.1) -> None:
        """
        Initilize weights for the ML.

        Parameters
        ----------
        passive_sd : float, optional
            Standard deviation around 0 for the inital phases. The default is 0.1.

        Returns
        -------
        None
            DESCRIPTION.

        """
        
        # Randomly set the phases (start parameter):
        phi_in = tf.random.normal(shape=[int((self.depth + (self.modes/2)) * self.layers)], stddev=passive_sd)
        phi_ex = tf.random.normal(shape=[int((self.depth + (self.modes/2)) * self.layers)], stddev=passive_sd)
        
        # Assigning the phases as the weights in our optimization systhem:
        self.weights = tf.convert_to_tensor([phi_in, phi_ex])
        
        self.weights = tf.Variable(tf.transpose(self.weights))
        
    def init_ML(self, target_state: list, fail_states: list, preparation: Optional[Callable], passive_sd: float = 0.1) -> None :
        """
        Initialises systhem for machine learning

        Parameters
        ----------
        target_state : list
            List of tuple containing the target states in the truncated Fock Space in order of the given input states (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
        fail_states : list
            List of tuple containing the fail states in the truncated Fock Space (eg. [(1,1,0,0),(2,0,0,0),...] == [|1100>, |2000>, ...]).
            Hereby is a fail state defined as a state in computational modes which do not represent a successful gate opertaion - keyword: photon bunching
        preparation : Optional[Callable]
            Strawberry fields function, which is applied directly after the initialisation, for example to generate ancilla states in superposition.
        passive_sd : float, optional
            Standard deviation around 0 for the inital phases. The default is 0.1.

        Returns
        -------
        None.

        """
        
        # Defining the quantum circuit enviroment for Strawberryfields:
        self.prog = sf.Program(self.modes)
        
        self.init_weights()
        names = ["phi_in","phi_ex"]
        
        # Assigning the weights as phases to the quantum circuit:
        self.sf_params = []
        
        for i in range(int((self.depth + (self.modes/2))*self.layers)):
            # For the ith layer, generate parameter names "phi_in_i", "phi_ex_i".
            # Also using phi_in and phi_ex for roation gates
            sf_params_names = ["{}_{}".format(n, i) for n in names]
            # Create the parameters, and append them to our list ``sf_params``.
            self.sf_params.append(self.prog.params(*sf_params_names))
            
        self.sf_params = np.array(self.sf_params)
        
        # Defining the actual quantum circuit:
        self.LinearOptics(preparation)
        
        # Defining the quatnum engine used for the calculation (Strawberryfields):
        self.eng = sf.Engine('tf', backend_options={"cutoff_dim": self.cutoff_dim, "batch_size": self.gate_cutoff})
        
        # Creating Masks for target_state and fail_states
        # This is necessary due to limitations of Tensorflow regarding advanced slicing in higher dimensions.
        #
        # TODO: Suggest fix to Tensorflow
        #
        if self.post_select is not None:
            if len(self.post_select) == 3:
                target_state = PostSelect(target_state, self.post_select[0], self.post_select[1], self.post_select[2])
                self.t_mask = FockBasis(target_state, self.cutoff_dim, self.modes, dtype = bool, multiplex = self.post_select[2])
            else:
                target_state = PostSelect(target_state, self.post_select[0], self.post_select[1])
                self.t_mask = FockBasis(target_state, self.cutoff_dim, self.modes, dtype = bool)
        else:
            self.t_mask = FockBasis(target_state, self.cutoff_dim, self.modes, dtype = bool)
        
        if fail_states is not None:
            if self.post_select is not None:
                if len(self.post_select) == 3:
                    fail_states = PostSelect(fail_states, self.post_select[0], self.post_select[1], self.post_select[2])
                    self.fail_mask = FockBasis(fail_states, self.cutoff_dim, self.modes, dtype = bool, multiplex = self.post_select[2])
                else:
                    fail_states = PostSelect(fail_states, self.post_select[0], self.post_select[1])
                    self.fail_mask = FockBasis(fail_states, self.cutoff_dim, self.modes, dtype = bool)
            else:
                self.fail_mask = FockBasis(fail_states, self.cutoff_dim, self.modes, dtype = bool)
        else:
            self.fail_mask = None
        
        # Update Machine Learning preparation flag:
        self.ML_initialised = True
    
    def LinearOptics(self, preparation: Optional[Callable], ML: bool = True, include_ket: bool = True):
        """
        Creates Linear Optical Systhem to systhesis the given gate
        Parameters
        ----------
        preparation : Optional[Callable]
            Strawberry fields function, which is applied directly after the initialisation, for example to generate ancilla states in superposition.
        ML : bool, optional
            States if the provided systhem will be used for training or evaluating. The default is True (= training).
        include_ket : bool, optional
            States if the input state should be included or not. The default is True.

        Returns
        -------
        None.

        """
        if ML:
            with self.prog.context as q:
                
                if include_ket:
                    ops.Ket(self.init_state) | q
                    
                if preparation is not None:
                    preparation() | q
                
                for layer in range(self.layers):
                    
                    if self.decomposition == 'Clements':
                        for i in range(int(self.modes/2)):
                            ops.Rgate(self.sf_params[int((layer * (self.depth + (self.modes/2))) + i + self.depth)][0]) | q[i * 2]
                            ops.Rgate(self.sf_params[int((layer * (self.depth + (self.modes/2))) + i + self.depth)][1]) | q[i * 2 + 1]
                            
                        for k in range(self.depth):
                            ops.MZgate(self.sf_params[int((layer * (self.depth + (self.modes/2))) + k)][0],
                                       self.sf_params[int((layer * (self.depth + (self.modes/2))) + k)][1]) | (q[int(self.struct[k])], q[int(self.struct[k] + 1)])
                            
                    elif self.decomposition == 'Reck':
                        parameter_index = list(range(int(self.depth + (self.modes/2))))
                        
                        for i in reversed(range(1, self.modes)):
                            for j in reversed(range(i)):
                                k = parameter_index.pop()
                                
                                ops.MZgate(self.sf_params[int((layer * (self.depth + (self.modes/2))) + k)][0],
                                           self.sf_params[int((layer * (self.depth + (self.modes/2))) + k)][1]).H | (q[i], q[j])
                                
                        for k in parameter_index:
                            ops.Rgate(-self.sf_params[int((layer * (self.depth + (self.modes/2))) + k)][0]) | q[k]
                            ops.Rgate(-self.sf_params[int((layer * (self.depth + (self.modes/2))) + k)][1]) | q[int(k + (self.modes/2))]
                            
        else:
            with self.prog_eval.context as q:
                
                if include_ket:
                    ops.Ket(self.init_state) | q
                    
                if preparation is not None:
                    preparation() | q
                    
                for layer in range(self.layers):
                    
                    if self.decomposition == 'Clements':
                        for i in range(int(self.modes/2)):
                            ops.Rgate(self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + i + self.depth)][0]) | q[i * 2]
                            ops.Rgate(self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + i + self.depth)][1]) | q[i * 2 + 1]
                            
                        for k in range(self.depth):
                            ops.MZgate(self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + k)][0],
                                       self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + k)][1]) | (q[int(self.struct[k])], q[int(self.struct[k] + 1)])
                            
                    elif self.decomposition == 'Reck':
                        parameter_index = list(range(int(self.depth + (self.modes/2))))
                        
                        for i in reversed(range(1, self.modes)):
                            for j in reversed(range(i)):
                                k = parameter_index.pop()
                                
                                ops.MZgate(self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + k)][0],
                                           self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + k)][1]).H | (q[i], q[j])
                                
                        for k in parameter_index:
                            ops.Rgate(-self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + k)][0]) | q[k]
                            ops.Rgate(-self.weights.numpy()[int((layer * (self.depth + (self.modes/2))) + k)][1]) | q[int(k + (self.modes/2))]
                            
    def cost(self, p_success: Optional[float], cost_factor: float, norm: list, punish: float) -> (float, list):
        """
        Costfunction for the Gradient Decent Algorithm

        Parameters
        ----------
        p_success : Optional[float]
            Set specific probabilty of success.
        cost_factor : float
            Factor to adjust the resulting cost function.
        norm : list
            List of type of norms to use in the cost function.
        punish : float
            Punishment of fail_states.

        Returns
        -------
        (float, list)
            Returns cost and the overlap of every single input case.

        """
        # Create a dictionary mapping from the names of the Strawberry Fields
        # free parameters to the TensorFlow weight values.
        mapping = {p.name: w for p, w in zip(self.sf_params.flatten(), tf.reshape(self.weights, [-1]))}
        
        # Run engine
        #eng.run(prog, args=mapping)
        #eng.print_applied()
        state = self.eng.run(self.prog, args=mapping).state

        # Extract the statevector
        ket = state.ket()
        
        A = tf.einsum('ijk -> kij',tf.stack([[tf.boolean_mask(ket, self.t_mask[i], axis = 1)[j] for j in range(self.gate_cutoff)] for i in range(self.t_mask.shape[0])]))
        
        if self.fail_state is not None and p_success is None:
            B = tf.einsum('ijk -> kij',tf.stack([[tf.boolean_mask(ket, self.fail_mask[i], axis = 1)[j] for j in range(self.gate_cutoff)] for i in range(self.fail_mask.shape[0])]))
            
        
        overlaps = tf.constant([0]*A.shape[1], dtype = float)
        if p_success is None:
            if norm[2] == 'mean':
                cost = cost_factor
            else:
                cost = cost_factor * tf.math.real(tf.math.pow(tf.norm(tf.ones_like(tf.linalg.diag_part(A[0])), norm[2]), 2))
        else:
            cost = 0
        for i in range(A.shape[0]): 
            if p_success is None:
                diag = tf.linalg.diag_part(A[i])
                off_diag = tf.math.real(tf.math.pow(tf.norm(tf.linalg.set_diag(A[i], [0]*A.shape[1]), norm[0]), 2))

                overlaps += tf.math.pow(tf.abs(diag), 2)
                if norm[2] == 'mean':
                    cost += tf.math.scalar_mul(-cost_factor, tf.math.real(tf.math.pow(tf.math.reduce_mean(diag) - tf.cast(tf.math.reduce_std(diag),np.complex64), 2))) + off_diag
                else:
                    cost += tf.math.scalar_mul(-cost_factor, tf.math.real(tf.math.pow(tf.norm(diag, norm[2]), 2))) + off_diag
                
                if self.fail_state is not None:
                    b = tf.math.real(tf.math.pow(tf.norm(B[i], norm[1]), 2))
                    cost += tf.math.scalar_mul(punish, b)
                '''
                    if off_diag == 0 and tf.math.count_nonzero(diag) == A.shape[1] and b == 0:
                        cost += - cost_factor / A.shape[0]
                    
                else:
                    if off_diag == 0 and tf.math.count_nonzero(diag) == A.shape[1]:
                        cost += - cost_factor / A.shape[0]
                '''
            else:
                overlaps += tf.math.pow(tf.abs(tf.linalg.diag_part(A[i])),2)
                cost += tf.math.scalar_mul(cost_factor, tf.math.real(tf.math.pow(tf.norm(A[i] - tf.cast(tf.linalg.diag([np.sqrt(p_success)]*self.gate_cutoff), np.complex64), norm[0]), 2)))
            
        if self.fail_mask is not None and p_success is not None:
            # Reduce fail_mask to relevant ancilla states (non-zero value after filtering with t_mask) 
            adaptive_fail_mask = tf.stack([[tf.boolean_mask(ket, self.t_mask[i], axis = 1)[j] for j in range(self.gate_cutoff)] for i in range(self.gate_cutoff)], axis=1)
            adaptive_fail_mask = tf.not_equal(tf.einsum('ijk->k', adaptive_fail_mask), tf.constant(0, dtype=np.complex64))
            
            B = tf.einsum('ijk -> kij',tf.boolean_mask(tf.stack([[tf.boolean_mask(ket, self.fail_mask[i], axis = 1)[j] for j in range(self.gate_cutoff)] for i in range(self.fail_mask.shape[0])]), adaptive_fail_mask, axis = 2))
            
            for i in range(B.shape[0]):
                cost += tf.math.scalar_mul(punish, tf.math.real(tf.math.pow(tf.norm(B[i], norm[1]), 2)))
            
        del ket

        return cost, overlaps
    
    #TODO
    def training_step(self, steps: int, optimizer: tf.keras.optimizers.Optimizer, p_success: float, cost_factor: float, norm: list, punish: float, early_stopping: bool, delta_step: float, conv_crit: int, auto_saving: bool, path: str, silence: bool) -> float:
        
        with tqdm.tqdm(range(1,steps+1), disable = silence) as pbar:
            convergence = 0
            for i in pbar:
                # reset the engine if it has already been executed
                if self.eng.run_progs:
                    self.eng.reset()
    
                # one repetition of the optimization
                with tf.GradientTape() as tape:
                    loss, overlaps_val = self.cost(p_success, cost_factor, norm, punish)
                    

                # calculate the mean overlap
                # This gives us an idea of how the optimization is progressing
                mean_overlap_val = np.mean(overlaps_val)
    
                # store cost at each step
                self.cost_progress.append(float(loss))
                self.overlap_progress.append(overlaps_val)
    
                # one repetition of the optimization
                gradients = tape.gradient(loss, self.weights)
                #print(loss)
                #print(gradients)
                optimizer.apply_gradients(zip([gradients], [self.weights]))
            
                pbar.set_postfix({'Cost':float(loss), 'Mean overlap':mean_overlap_val})
                
                if early_stopping and loss < delta_step:
                    break
                
                if early_stopping and i > conv_crit:
                    if abs(self.cost_progress[-2] - loss) < delta_step:
                        convergence += 1
                        if convergence > conv_crit:
                            break
                    else:
                        convergence = 0
                        
                if path is not None:
                    if auto_saving > 0:
                        if i % auto_saving == 0:
                            self.save(path)
                            
        return loss, mean_overlap_val
        
    
    def fit(self, target_state: list, fail_states: Optional[list] = None, preparation: Optional[Callable] = None, post_select: Optional[list] = None, steps: int = 500, repeat: Optional[int] = None, optimizer: tf.keras.optimizers.Optimizer = None, learning_rate: float = 0.025, punish: float = 0.1, cost_factor: float = 1, norm: list = [2, 2, 'mean'], conv_crit: int = 50, delta_step: float = 1e-6, early_stopping: bool = True, n_sweeps: Optional[int] = None, sweep_low: float = 0.0, sweep_high: float = 1.0, p_success: Optional[float] = None, path: Optional[str] = None, auto_saving: int = 100, load: Union[bool, str] = True, silence: bool = False):
        """
        Adjusting the weights based on a gradient decent algorithm.

        Parameters
        ----------
        target_state : list
            List of tuple containing the target states in the truncated Fock Space in order of the given input states (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
        fail_states : Optional[list], optional
            List of tuple containing the fail states in the truncated Fock Space (eg. [(1,1,0,0),(2,0,0,0),...] == [|1100>, |2000>, ...]).
            Hereby is a fail state defined as a state in computational modes which do not represent a successful gate opertaion - keyword: photon bunching
            The default is None.
        preparation : Optional[Callable], optional
            Strawberry fields function, which is applied directly after the initialisation, for example to generate ancilla states in superposition.
            The default is None.
        post_select : Optional[list], optional
            More detailed definition of the ancilla modes. The default is None.
        steps : int, optional
            Number of optimization steps to perform. The default is 500.
        repeat : Optional[int]
            Repeat traing x times. The default is None.
        optimizer : tf.keras.optimizers.Optimizer, optional
            Optimizer to use for the gradient decent algorithm. The default is None.
        learning_rate : float, optional
            Step length for the optimizer. The default is 0.025.
        punish : float, optional
            Variable for the contribution of fail_states to the cost function. The default is 0.1.
        cost_factor : float, optional
            Variable for the scaling of the cost function excluding fail states. The default is 1.
        norm : list, optional
            Norms to be used for the main path [0], the fail states [1] and the diagonal of the main path [2]. The default is [2, 2, 'mean'].
        conv_crit : int, optional
            Number of consequent steps, which induced a change smaller than delt_step, which are necessary to trigger early stopping. The default is 50.
        delta_step : float, optional
            Minimum change necessary to count as progress in convergence. The default is 1e-6.
        early_stopping : bool, optional
            Stops if sytem is converge before the maximal number of steps is made. The default is True.
        n_sweeps : Optional[int], optional
            Determin if a sweep over the possible possiblities of success should be made and the number of sweeps. The default is None.
        sweep_low : float, optional
            Starting point of the sweep operation. The default is 0.0.
        sweep_high : float, optional
            End point of the sweep operation. The default is 1.0.
        p_success : Optional[float], optional
            Set probability of success. The default is None.
        path : Optional[str], optional
            Destination, where data  aquired by the sweep operation should be stored. The default is None.
        auto_saving: int
            Number of iterations before the weights are saved, iff a path is provided. If set to 0 no auto_saves will be performed. The default is 100.
        load: Union[bool, str]
            Defines if preexisting results should be loaded. If True is passed the provided path will be used otherwise if a string is passed this file be loaded instead. The default is True.
        silence : bool, optional
            Silence outputs. The default is False.

        Returns
        -------
        None.

        """
       
        self.post_select = post_select
        self.target_state = target_state
        self.fail_states = fail_states
        
        if not self.ML_initialised:
            self.init_ML(target_state, fail_states, preparation)
            
        if optimizer is None:
            # Using SGD algorithm for optimization
            optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.025, momentum=0.9)
            
        #Get initial states of the optimizer and learning rate decay
        init_state_opt = optimizer.get_config()
        if type(learning_rate) != float:
            init_state_lr = learning_rate.get_config()
            
        # Run optimization
        if n_sweeps is None:
            if repeat is None:
                if type(load) == str:
                    self.load(load)
                    
                elif load == True and path is not None and (('.' in path and os.path.isfile(path)) or ('.' not in path and os.path.isfile(path+'.npz'))):
                    self.load(path)  
                
                self.overlap_progress = []
                self.cost_progress = []
    
                _ = self.training_step(steps, optimizer, p_success, cost_factor, norm, punish, early_stopping, delta_step, conv_crit, auto_saving, path, silence)
                                    
                if path is not None:
                     self.save(path)
            
            else:
                best_loss = np.inf
                best_weights = tf.Variable(self.weights)
                best_overlap_progress = []
                total_cost_progress = []
                
                for rep in range(repeat):
                    if type(load) == str:
                        self.load(load)
                        
                    elif load == True and path is not None and (('.' in path and os.path.isfile(path)) or ('.' not in path and os.path.isfile(path+'.npz'))):
                        self.load(path)
                        
                    else:
                        self.init_weights()
                    
                    self.overlap_progress = []
                    self.cost_progress = []
                    
                    if type(learning_rate) != float:
                        learning_rate = learning_rate.from_config(init_state_lr)
                    optimizer = optimizer.from_config(init_state_opt)
                    
                    if path is None:
                        file = None
                    else:
                        fiel = path+'_try_'+str(rep)
        
                    loss, _ = self.training_step(steps, optimizer, p_success, cost_factor, norm, punish, early_stopping, delta_step, conv_crit, auto_saving, file, silence)
                    
                    total_cost_progress.append(np.copy(self.cost_progress))
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_weights = tf.Variable(self.weights)
                        best_overlap_progress = np.copy(self.overlap_progress)
                        
                    if loss < delta_step:
                        break
                
                self.weights = best_weights
                self.overlap_process = best_overlap_progress
                self.cost_progress = total_cost_progress
                
                if path is not None:
                    self.save(path)
                 
        else:
            if path is not None and not os.path.exists(path):
                os.mkdir(path)
            cost_sweep = []
            
            if type(n_sweeps) == int:
                sweep = np.linspace(sweep_low, sweep_high, num = n_sweeps)  
            else:
                sweep = n_sweeps
                
            with tqdm.tqdm(sweep, disable = silence, position=0) as sweep_tqdm:
                for s in sweep_tqdm:
                    
                    if repeat is None:
                    
                        if type(load) == str:
                            self.load(load)
                        elif load == True and path is not None:
                            if ('.' in path and os.path.isfile(path)) or ('.' not in path and os.path.isfile(path+'.npz')):
                                self.load(path)
                            else:
                                file = path + '/' + str(round(s,len(str(steps+1)))).split('.')[1]
                                if os.path.isfile(file+'.npz'):
                                    self.load(file)
                            
                        self.overlap_progress = []
                        self.cost_progress = []
                        
                        if type(learning_rate) != float:
                            learning_rate = learning_rate.from_config(init_state_lr)
                        optimizer = optimizer.from_config(init_state_opt)
                        
                        if path is not None:
                            file = path + '/' + str(round(s,len(str(steps+1)))).split('.')[1]
                        else:
                            file = None
                        
                        loss, mean_overlap_val = self.training_step(steps, optimizer, s, cost_factor, norm, punish, early_stopping, delta_step, conv_crit, auto_saving, file, True)
                        
                        cost_sweep.append(loss)
                        sweep_tqdm.set_postfix({'Sweep':float(s), 'Cost':float(loss), 'Overlap':float(mean_overlap_val)})
                        
                        if path is not None:
                            self.save(file)
                            if fail_states is None:
                                self.visualize(path = file, title = s)
                            else:
                                self.visualize(path = file, title = s)
                            
                            self.evaluate(target_state = self.target_state, fail_states = self.fail_states, post_select = self.post_select, path = file)
                            
                    else:
                        best_loss = np.inf
                        best_weights = tf.Variable(self.weights)
                        best_overlap_progress = []
                        best_mean_overlap_val = 0
                        total_cost_progress = []
                        
                        for rep in range(repeat):

                            if type(load) == str:
                                self.load(load)
                            elif load == True and path is not None and (('.' in path and os.path.isfile(path)) or ('.' not in path and os.path.isfile(path+'.npz'))):
                                self.load(path)
                            elif load == True and path is not None and (os.path.isfile(path + '/' + str(round(s,len(str(steps+1)))).split('.')[1]+'.npz')):
                                self.load(path + '/' + str(round(s,len(str(steps+1)))).split('.')[1])
                            else:
                                self.init_weights()
                                
                            self.overlap_progress = []
                            self.cost_progress = []
                            
                            if type(learning_rate) != float:
                                learning_rate = learning_rate.from_config(init_state_lr)
                            optimizer = optimizer.from_config(init_state_opt)
                            
                            if path is not None:
                                f = path + '/' + str(round(s,len(str(steps+1)))).split('.')[1]+'_try_'+str(rep)
                            else:
                                f = None
                            
                            loss, mean_overlap_val = self.training_step(steps, optimizer, s, cost_factor, norm, punish, early_stopping, delta_step, conv_crit, auto_saving, f, True)
                            
                            total_cost_progress.append(np.copy(self.cost_progress))
                            
                            if loss < best_loss:
                                best_loss = loss
                                best_weights = tf.Variable(self.weights)
                                best_overlap_progress = np.copy(self.overlap_progress)
                                best_mean_overlap_val = mean_overlap_val
                                
                            if loss < delta_step:
                                break

                        self.weights = best_weights
                        self.overlap_progress = best_overlap_progress
                        self.cost_progress = total_cost_progress
                            
                        cost_sweep.append(best_loss)
                        sweep_tqdm.set_postfix({'Sweep':float(s), 'Cost':float(best_loss), 'Overlap':float(best_mean_overlap_val)})
                        
                        if path is not None:
                            file = path + '/' + str(round(s,len(str(steps+1)))).split('.')[1]
                            self.save(file)
                            self.visualize(path = file, title = s)
                            self.evaluate(target_state = self.target_state, fail_states = self.fail_states, post_select = self.post_select, path = file)
                            

            fig_sweep, ax_sweep = plt.subplots()
            fig_sweep.suptitle('p_success sweep')
            #ax_sweep.plot(sweep,cost_sweep,marker='o',markevery=5)
            ax_sweep.plot(sweep,cost_sweep)
            ax_sweep.set_ylabel('Cost')
            ax_sweep.set_xlabel('p_success')
            
            if path is not None:
                plt.savefig(path+'/Sweep')
                plt.close(fig_sweep)
                np.save(path+'/sweep',cost_sweep)
            
            cost_min = min(reversed(cost_sweep))
            print('Minimal of the cost function: ', float(cost_min))
            print('at a success probability of: ',sweep[cost_sweep.index(cost_min)])
            
    def visualize(self, path: Optional[str] = None, title: Optional[str] = None, legend: str = 'CNOT'):
        """
        Visualisation of the training of the machine learning model.

        Parameters
        ----------
        path : Optional[str], optional
            Path to save the plots. The default is None.
        title : Optional[str], optional
            Title of the plots. The default is None.
        legend : str, optional
           Legend to the plots. The default is 'CNOT'.

        Returns
        -------
        None.

        """
        legend_dict = {'CZ':['|HH> -> |HH>', '|HV> -> |HV>', '|VH> -> |VH>', '|VV> -> -|VV>'],
                       'CNOT':['|HH> -> |HH>', '|HV> -> |HV>', '|VH> -> |VV>', '|VV> -> |VH>'],
                       '+/- CNOT':['|++> -> |++>','|+-> -> |-->','|-+> -> |-+>','|--> -> |+->'],
                       'complete CNOT': ['|++> -> |++>','|+-> -> |-->','|-+> -> |-+>','|--> -> |+->',
                                         '|HH> -> |HH>', '|HV> -> |HV>', '|VH> -> |VV>', '|VV> -> |VH>']}
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
        plt.style.use('default')

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        if title is not None:
            fig.suptitle(title)
            
        if type(self.cost_progress[0]) == np.ndarray:
            [ax[0].plot(self.cost_progress[i]) for i in range(len(self.cost_progress))]
        else:
            ax[0].plot(self.cost_progress)
            
        ax[0].set_ylabel('Cost')
        ax[0].set_xlabel('Step')
        ax[0].set_yscale('log')
        
        ax[1].plot(self.overlap_progress)
        ax[1].yaxis.set_label_position("right")
        ax[1].yaxis.tick_right()
        ax[1].set_ylabel('Overlap')
        ax[1].set_xlabel('Step')
        ax[1].legend(legend_dict[legend])
        if path is not None:
            plt.savefig(path+'_fit')
        else:
            plt.show()
            
        plt.close(fig)
        
    def save(self, path: str):
        """
        Save QGS object.

        Parameters
        ----------
        path : str
            Path to save file location.

        Returns
        -------
        None.

        """
        np.savez(path, photons = self.photons, cutoff_dim = self.cutoff_dim, modes = self.modes, depth = self.depth, layers = self.layers, decomposition = self.decomposition, struct = self.struct, init_state = self.init_state, weights = self.weights)
        
    def load(self, path: str):
        """
        

        Parameters
        ----------
        path : str
            Path to save file location.

        Returns
        -------
        None.

        """
        if '.' not in path:
            path += '.npz'
        loaded_QGS = np.load(path)
        self.photons = int(loaded_QGS['photons'])
        self.cutoff_dim = int(loaded_QGS['cutoff_dim'])
        self.modes = int(loaded_QGS['modes'])
        self.depth = int(loaded_QGS['depth'])
        self.layers = int(loaded_QGS['layers'])
        self.decomposition = loaded_QGS['decomposition'] 
        self.struct = list(loaded_QGS['struct'])
        self.init_state = tf.constant(loaded_QGS['init_state'])
        self.weights = tf.Variable(loaded_QGS['weights'])
        
    def evaluate(self, target_state: Optional[list] = None, fail_states: Optional[list] = None, preparation: Optional[Callable] = None, post_select: Optional[list] = None, ket: Optional[np.ndarray] = None, path: Optional[str] = None, precision: int = 3, title: Optional[str] = None, permutation: list = [0,1,3,2], verbosity: int = 2, return_ket = False) -> np.ndarray:
        """
        Evaluate given sythesised gate.

        Parameters
        ----------
        target_state : Optional[list], optional
            List of tuple containing the target states in the truncated Fock Space in order of the given input states (eg. [(1,0,1,0),(1,0,0,1),...] == [|1010>, |1001>, ...])
            The default is None.
        fail_states : Optional[list], optional
            List of tuple containing the fail states in the truncated Fock Space (eg. [(1,1,0,0),(2,0,0,0),...] == [|1100>, |2000>, ...]).
            Hereby is a fail state defined as a state in computational modes which do not represent a successful gate opertaion - keyword: photon bunching
            The default is None.
        preparation : Optional[Callable], optional
            Strawberry fields function, which is applied directly after the initialisation, for example to generate ancilla states in superposition.
            The default is None.
        post_select : Optional[list], optional
            More detailed definition of the ancilla modes. The default is None.
        ket : Optional[np.ndarray], optional
            Provid already calulated ket state for evaluation and skip so the calulation
            The default is None.
        path : Optional[str], optional
            Path for saving created plots and reports. The default is None.
        precision : int, optional
            Number of digits after the decimal point to print. The default is 3.
        title : Optional[str], optional
            Titel for the created plots. The default is None.
        permutation : list, optional
            Permutation between input and target states. The default is [0,1,3,2] = CNOT.
        verbosity : int, optional
            Set verbosity for amount of information in the report:
            verbosity = 2 : All information in the report
                        1 : Only post selected states and target state propapility
                        0 : No data reported only visualization
            The default is 2.
        return_ket : bool, optional
            Define if ket should be returned or deleted.
            The default is False.

        Returns
        -------
        Plots and evaluation report

        """   
        self.path = path
        self.print_Circuit(preparation)
        
        self.prog_eval = sf.Program(self.modes)
        self.LinearOptics(preparation, False)
        
        return evaluate(self.prog_eval, target_state = target_state, fail_states = fail_states, post_select = post_select, ket = ket, cutoff_dim = self.cutoff_dim, gate_cutoff = self.gate_cutoff, path = path, precision = precision, title = title, permutation = permutation, verbosity = verbosity, return_ket = return_ket)
        
    def print_Circuit(self, preparation: Optional[Callable], optimize: bool = False):
        """
        Prints linear Optical circuit

        Parameters
        ----------
        preparation : Optional[Callable]
            Strawberry fields function, which is applied directly after the initialisation, for example to generate ancilla states in superposition.
            The default is None.
        optimize : TYPE, optional
            Toogles if Strawberryfields should try to optimize the circuit. The default is False.

        Returns
        -------
        None.

        """
        self.prog_eval = sf.Program(self.modes)
        
        self.LinearOptics(preparation, ML=False, include_ket=False)
        if optimize:
            self.prog_eval.optimize()
        self.prog_eval.print(self.print)
        
    def print(self, *args, **kwargs):
        """
        Either prints to console or to a report file.

        """
        if self.path is None:
            print(*args, **kwargs)
        else:
            with open(self.path + '.txt', 'a') as f:
                with redirect_stdout(f):
                    print(*args, **kwargs)