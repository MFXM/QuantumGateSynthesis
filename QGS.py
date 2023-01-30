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
    
def FockBasis(state_list: list, cutoff_dim: int, modes: int, dtype: type = np.complex64) -> np.ndarray: 
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
    
    Returns
    -------
    numpy.ndarray
        complete numpy array to be handeld by Strawberryfields
        
    """
    state = []
    for _ in range(len(state_list)):
        state.append(np.zeros([cutoff_dim]*modes, dtype=dtype))
    
    for i in range(len(state_list)):
        state[i][state_list[i]] = 1
    
    return np.array(state)

def PostSelect(state_list: list, ps: list, mask: Optional[list] = None) -> list:
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

    Returns
    -------
    list
        New list of states, where each state now also contains information about the ancilla qubits

    """
    state_l = state_list.copy()
    
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
                
def evaluate(circuit: sf.Program, target_state: Optional[list] = None, fail_states: Optional[list] = None, post_select: Optional[list] = None, backend: str = 'tf', cutoff_dim: int = 3, gate_cutoff: int = 4, path: Optional[str] = None, precision: int = 3, title: Optional[str] = None, permutation: list = [0,1,3,2]):
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

    Returns
    -------
    Plots and evaluation report

    """
    
    eng = sf.Engine(backend, backend_options={"cutoff_dim": cutoff_dim, "batch_size": gate_cutoff})
        
    if target_state is not None:
        if post_select is not None:
            target_state = PostSelect(target_state, post_select[0], post_select[1])
        t_mask = FockBasis(target_state, cutoff_dim, circuit.num_subsystems, dtype = bool)
    
    if fail_states is not None:
        if post_select is not None:
                fail_states = PostSelect(fail_states, post_select[0], post_select[1])
        fail_mask = FockBasis(fail_states, cutoff_dim, circuit.num_subsystems, dtype = bool)
        
    result = eng.run(circuit)
    
    state = result.state
    
    ket = state.ket()
    
    if tf.is_tensor(ket):
        ket = ket.numpy()
        
    # Check the normalization of the ket.
    # This does give the exact answer because of the cutoff we chose.
    for i in range(ket.shape[0]):
        report(path, f"\n{i+1}. Input:\n")        
        
        report(path, "The norm of the ket is ", np.linalg.norm(ket[i]))
        
        ket_rounded = np.round(ket[i], precision)
        ind = np.array(np.nonzero(np.real_if_close(ket_rounded))).T
        # And these are their coefficients
        
        report(path, "\nThe nonzero components are:")
        for state in ind:
            report(path, ket_rounded[tuple(state)], tuple(state))
            
        if target_state is not None:
            target_ket = np.round(tf.boolean_mask(ket, t_mask[i], axis = 1)[i], precision)
            report(path, "\nTarget state:",target_state[i])
            p_target_state = np.round(np.linalg.norm(target_ket) ** 2, precision*2)
            report(path,"The overall propability to get the target state is", p_target_state)
        
    if target_state is not None:   
                
        A = tf.stack([[tf.math.reduce_sum(tf.boolean_mask(ket, t_mask[i], axis = 1)[j]) for j in range(gate_cutoff)] for i in range(gate_cutoff)], axis=1).numpy()[:,permutation]
        
        if fail_states is not None:
            B = tf.stack([[tf.math.reduce_sum(tf.boolean_mask(ket, fail_mask[i], axis = 1)[j]) for j in range(gate_cutoff)] for i in range(fail_mask.shape[0])],axis=1).numpy()
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
        
        state_label = ['|HH>','|HV>','|VH>','|VV>']
        f_state_label = ['|1100>', '|0011>', '|2000>', '|0200>', '|0020>', '|0002>']
        
        xpos = np.arange(0,4,1)
        ypos = np.arange(0,4,1)
        xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)
        
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(4*4)
        
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = np.absolute(A.flatten())
        
        cs = [[c]*4 for c in ['r','g','b','c']]
        cs = list(np.array(cs).flatten())
        
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=cs)
        ax1.set_xticks([0.5,1.5,2.5,3.5])
        ax1.w_xaxis.set_ticklabels(state_label)
        ax1.set_yticks([0.5,1.5,2.5,3.5])
        ax1.w_yaxis.set_ticklabels(state_label)
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
            
    del ket
            
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
        
    def init_ML(self, target_state: list, fail_states: list, preperation: Optional[Callable], passive_sd: float = 0.1) -> None :
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
        
        # Randomly set the phases (start parameter):
        phi_in = tf.random.normal(shape=[int((self.depth + (self.modes/2)) * self.layers)], stddev=passive_sd)
        phi_ex = tf.random.normal(shape=[int((self.depth + (self.modes/2)) * self.layers)], stddev=passive_sd)
        
        # Assigning the phases as the weights in our optimization systhem:
        self.weights = tf.convert_to_tensor([phi_in, phi_ex])
        names = ["phi_in","phi_ex"]
        self.weights = tf.Variable(tf.transpose(self.weights))
        
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
        self.LinearOptics(preperation)
        
        # Defining the quatnum engine used for the calculation (Strawberryfields):
        self.eng = sf.Engine('tf', backend_options={"cutoff_dim": self.cutoff_dim, "batch_size": self.gate_cutoff})
        
        # Creating Masks for target_state and fail_states
        # This is necessary due to limitations of Tensorflow regarding advanced slicing in higher dimensions.
        #
        # TODO: Suggest fix to Tensorflow
        #
        if self.post_select is not None:
            target_state = PostSelect(target_state, self.post_select[0], self.post_select[1])
        self.t_mask = FockBasis(target_state, self.cutoff_dim, self.modes, dtype = bool)
        
        if fail_states is not None:
            if self.post_select is not None:
                fail_states = PostSelect(fail_states, self.post_select[0], self.post_select[1])
            self.fail_mask = FockBasis(fail_states, self.cutoff_dim, self.modes, dtype = bool)
        else:
            self.fail_mask = None
        
        # Update Machine Learning preparation flag:
        self.ML_initialised = True
    
    def LinearOptics(self, preperation: Optional[Callable], ML: bool = True, include_ket: bool = True):
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
                    
                if preperation is not None:
                    preperation() | q
                
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
                    
                if preperation is not None:
                    preperation() | q
                    
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
                            
    def cost(self, p_success: float, cost_factor: float, norm: list, punish: float) -> (float, list):
        """
        Costfunction for the Gradient Decent Algorithm

        Parameters
        ----------
        p_success : float
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
        
        self.A = tf.stack([[tf.math.reduce_sum(tf.boolean_mask(ket, self.t_mask[i], axis = 1)[j]) for j in range(self.gate_cutoff)] for i in range(self.gate_cutoff)], axis=1)
        
        overlaps = tf.math.pow(tf.abs(tf.linalg.diag_part(self.A)),2)
        cost = tf.math.scalar_mul(cost_factor, tf.math.real(tf.math.pow(tf.norm(self.A - tf.cast(tf.linalg.diag([np.sqrt(p_success)]*self.gate_cutoff), np.complex64), norm[0]), 2)))
    
        if self.fail_mask is not None:
            # Reduce fail_mask to relevant ancilla states (non-zero value after filtering with t_mask) 
            adaptive_fail_mask = tf.stack([[tf.boolean_mask(ket, self.t_mask[i], axis = 1)[j] for j in range(self.gate_cutoff)] for i in range(self.gate_cutoff)], axis=1)
            adaptive_fail_mask = tf.not_equal(tf.einsum('ijk->k', adaptive_fail_mask), tf.constant(0, dtype=np.complex64))
            
            self.B = tf.einsum('ijk -> ij',tf.boolean_mask(tf.stack([[tf.boolean_mask(ket, self.fail_mask[i], axis = 1)[j] for j in range(self.gate_cutoff)] for i in range(self.fail_mask.shape[0])], axis = 1), adaptive_fail_mask, axis = 2))
            
            cost += tf.math.scalar_mul(punish, tf.math.real(tf.math.pow(tf.norm(self.B, norm[1]), 2)))
            
        if len(norm) == 3:
            cost += tf.math.real(tf.math.pow(tf.norm(self.A - tf.cast(tf.linalg.diag([np.sqrt(p_success)]*self.gate_cutoff), np.complex64), norm[2]), 2))
        
        del ket

        return cost, overlaps
    
    def fit(self, target_state: list, fail_states: Optional[list] = None, preparation: Optional[Callable] = None, post_select: Optional[list] = None, reps: int = 500, opt: tf.keras.optimizers.Optimizer = None, learning_rate: float = 0.025, punish: float = 0.1, cost_factor: float = 1, norm: list = [2, np.inf], conv_crit: int = 50, delta_step: float = 1e-6, early_stopping: bool = True, n_sweeps: Optional[int] = None, sweep_low: float = 0.0, sweep_high: float = 1.0, p_success: float = 1.0, path: Optional[str] = None, auto_saving: int = 100, load: Union[bool, str] = True, silence: bool = False):
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
        reps : int, optional
            Number of optimization steps to perform. The default is 500.
        opt : tf.keras.optimizers.Optimizer, optional
            Optimizer to use for the gradient decent algorithm. The default is None.
        learning_rate : float, optional
            Step length for the optimizer. The default is 0.025.
        punish : float, optional
            Variable for the contribution of fail_states to the cost function. The default is 0.1.
        cost_factor : float, optional
            Variable for the scaling of the cost function excluding fail states. The default is 1.
        norm : list, optional
            Norms to be used for the main path [0] and the fail states [1]. The default is [2, np.inf].
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
        p_success : float, optional
            Set probability of success. The default is 1.0.
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

        convergence = 0
        
        self.post_select = post_select
        self.target_state = target_state
        self.fail_states = fail_states
        
        if not self.ML_initialised:
            self.init_ML(target_state, fail_states, preparation)
            
        if opt is None:
            # Using Nadam algorithm for optimization
            opt = tf.keras.optimizers.Nadam(learning_rate = learning_rate)
            
        # Run optimization
        if n_sweeps is None:
            if type(load) == str:
                self.load(load)
                
            elif load == True and path is not None:
                if ('.' in path and os.path.isfile(path)) or ('.' not in path and os.path.isfile(path+'.npz')):
                    self.load(path)
            
            self.overlap_progress = []
            self.cost_progress = []
            with tqdm.tqdm(range(1,reps+1), disable = silence) as pbar:
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
                    opt.apply_gradients(zip([gradients], [self.weights]))
                
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
                                
                if path is not None:
                     self.save(path)
        
        else:
            if path is not None and not os.path.exists(path):
                os.mkdir(path)
            cost_sweep = []
            sweep = np.linspace(sweep_low, sweep_high, num = n_sweeps)   
            with tqdm.tqdm(sweep, disable = silence, position=0) as sweep:
                for s in sweep:
                    
                    if type(load) == str:
                        self.load(load)
                    elif load == True and path is not None:
                        if ('.' in path and os.path.isfile(path)) or ('.' not in path and os.path.isfile(path+'.npz')):
                            self.load(path)
                        else:
                            file = path + '/' + str(round(s,len(str(reps+1)))).split('.')[1]
                            if os.path.isfile(file+'.npz'):
                                self.load(file)
                        
                    self.overlap_progress = []
                    self.cost_progress = []
                    
                    with tqdm.tqdm(range(1,reps+1), disable = True, position=1, leave = False) as pbar:
                        for i in pbar:
                            # reset the engine if it has already been executed
                            if self.eng.run_progs:
                                self.eng.reset()
                
                            # one repetition of the optimization
                            with tf.GradientTape() as tape:
                                loss, overlaps_val = self.cost(s, cost_factor, norm, punish)
                                
            
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
                            opt.apply_gradients(zip([gradients], [self.weights]))
                            
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
                                        file = path + '/' + str(round(s,len(str(reps+1)))).split('.')[1]
                                        self.save(file)
                                        
                            pbar.set_postfix({'Cost':float(loss), 'Mean overlap':mean_overlap_val})
                            
                    cost_sweep.append(loss)
                    sweep.set_postfix({'Sweep':float(s), 'Cost':float(loss), 'Overlap':float(mean_overlap_val)})
                    
                    if path is not None:
                        file = path + '/' + str(round(s,len(str(reps+1)))).split('.')[1]
                        self.save(file)
                        if fail_states is None:
                            self.visualize(path = file, title = s)
                        else:
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
        ax[0].plot(self.cost_progress)
        ax[0].set_ylabel('Cost')
        ax[0].set_xlabel('Step')
        
        ax[1].plot(self.overlap_progress)
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
        
    def evaluate(self, target_state: Optional[list] = None, fail_states: Optional[list] = None, preperation: Optional[Callable] = None, post_select: Optional[list] = None, path: Optional[str] = None, precision: int = 3, title: Optional[str] = None, permutation: list = [0,1,3,2]):
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
        path : Optional[str], optional
            Path for saving created plots and reports. The default is None.
        precision : int, optional
            Number of digits after the decimal point to print. The default is 3.
        title : Optional[str], optional
            Titel for the created plots. The default is None.
        permutation : list, optional
            Permutation between input and target states. The default is [0,1,3,2] = CNOT.

        Returns
        -------
        Plots and evaluation report

        """   
        self.path = path
        self.print_Circuit(preperation)
        
        self.prog_eval = sf.Program(self.modes)
        self.LinearOptics(preperation, False)
        
        evaluate(self.prog_eval, target_state = target_state, fail_states = fail_states, cutoff_dim = self.cutoff_dim, gate_cutoff = self.gate_cutoff, path = path, precision = precision, title = title, permutation = permutation)#
        
    def print_Circuit(self, preperation: Optional[Callable], optimize: bool = False):
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
        
        self.LinearOptics(preperation, ML=False, include_ket=False)
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