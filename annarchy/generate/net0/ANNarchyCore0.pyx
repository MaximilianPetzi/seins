# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from math import ceil
import numpy as np
import sys
cimport numpy as np

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport LILConnectivity as LIL
from ANNarchy.core.cython_ext.Connector cimport CSRConnectivity, CSRConnectivityPre1st

cdef extern from "ANNarchy.h":

    # User-defined functions


    # User-defined constants


    # Data structures

    # Export Population 0 (pop0)
    cdef struct PopStruct0 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Local parameter r
        vector[double] get_r()
        double get_single_r(int rk)
        void set_r(vector[double])
        void set_single_r(int, double)



        # Targets



        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 1 (pop1)
    cdef struct PopStruct1 :
        # Number of neurons
        int get_size()
        void set_size(int)
        # Maximum delay in steps
        int get_max_delay()
        void set_max_delay(int)
        void update_max_delay(int)
        # Activate/deactivate the population
        bool is_active()
        void set_active(bool)
        # Reset the population
        void reset()


        # Global parameter tau
        double  get_tau()
        void set_tau(double)

        # Local parameter constant
        vector[double] get_constant()
        double get_single_constant(int rk)
        void set_constant(vector[double])
        void set_single_constant(int, double)

        # Global parameter alpha
        double  get_alpha()
        void set_alpha(double)

        # Global parameter f
        double  get_f()
        void set_f(double)

        # Global parameter A
        double  get_A()
        void set_A(double)

        # Local variable perturbation
        vector[double] get_perturbation()
        double get_single_perturbation(int rk)
        void set_perturbation(vector[double])
        void set_single_perturbation(int, double)

        # Local variable noise
        vector[double] get_noise()
        double get_single_noise(int rk)
        void set_noise(vector[double])
        void set_single_noise(int, double)

        # Local variable x
        vector[double] get_x()
        double get_single_x(int rk)
        void set_x(vector[double])
        void set_single_x(int, double)

        # Local variable rprev
        vector[double] get_rprev()
        double get_single_rprev(int rk)
        void set_rprev(vector[double])
        void set_single_rprev(int, double)

        # Local variable r
        vector[double] get_r()
        double get_single_r(int rk)
        void set_r(vector[double])
        void set_single_r(int, double)

        # Local variable delta_x
        vector[double] get_delta_x()
        double get_single_delta_x(int rk)
        void set_delta_x(vector[double])
        void set_single_delta_x(int, double)

        # Local variable x_mean
        vector[double] get_x_mean()
        double get_single_x_mean(int rk)
        void set_x_mean(vector[double])
        void set_single_x_mean(int, double)



        # Targets
        vector[double] _sum_exc
        vector[double] _sum_in



        # memory management
        long int size_in_bytes()
        void clear()


    # Export Projection 0
    cdef struct ProjStruct0 :
        # Flags
        bool _transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset
        # Size
        int get_size()
        int nb_synapses(int)
        void set_size(int)


        # LIL Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        void inverse_connectivity_matrix()

        # Local variable w
        vector[vector[double]] get_w()
        vector[double] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[vector[double]])
        void set_dendrite_w(int, vector[double])
        void set_synapse_w(int, int, double)








        # memory management
        long int size_in_bytes()
        void clear()

    # Export Projection 1
    cdef struct ProjStruct1 :
        # Flags
        bool _transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset
        # Size
        int get_size()
        int nb_synapses(int)
        void set_size(int)


        # LIL Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        void inverse_connectivity_matrix()

        # Local variable w
        vector[vector[double]] get_w()
        vector[double] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[vector[double]])
        void set_dendrite_w(int, vector[double])
        void set_synapse_w(int, int, double)




        # Global parameter effective_eta
        double get_effective_eta()
        void set_effective_eta(double)

        # Global parameter learning_phase
        double get_learning_phase()
        void set_learning_phase(double)

        # Global parameter error
        double get_error()
        void set_error(double)

        # Global parameter mean_error
        double get_mean_error()
        void set_mean_error(double)

        # Global parameter mean_mean_error
        double get_mean_mean_error()
        void set_mean_mean_error(double)

        # Global parameter max_weight_change
        double get_max_weight_change()
        void set_max_weight_change(double)

        # Local variable trace
        vector[vector[double]] get_trace()
        vector[double] get_dendrite_trace(int)
        double get_synapse_trace(int, int)
        void set_trace(vector[vector[double]])
        void set_dendrite_trace(int, vector[double])
        void set_synapse_trace(int, int, double)

        # Local variable delta_w
        vector[vector[double]] get_delta_w()
        vector[double] get_dendrite_delta_w(int)
        double get_synapse_delta_w(int, int)
        void set_delta_w(vector[vector[double]])
        void set_dendrite_delta_w(int, vector[double])
        void set_synapse_delta_w(int, int, double)





        # memory management
        long int size_in_bytes()
        void clear()


    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period_
        int period_offset_
        long offset_

    void addRecorder(Monitor*)
    void removeRecorder(Monitor*)

    # Population 0 (pop0) : Monitor
    cdef cppclass PopRecorder0 (Monitor):
        PopRecorder0(vector[int], int, int, long) except +
        long int size_in_bytes()
        void clear()

        vector[vector[double]] r
        bool record_r

        # Targets
    # Population 1 (pop1) : Monitor
    cdef cppclass PopRecorder1 (Monitor):
        PopRecorder1(vector[int], int, int, long) except +
        long int size_in_bytes()
        void clear()

        vector[double] tau
        bool record_tau

        vector[vector[double]] constant
        bool record_constant

        vector[double] alpha
        bool record_alpha

        vector[double] f
        bool record_f

        vector[double] A
        bool record_A

        vector[vector[double]] perturbation
        bool record_perturbation

        vector[vector[double]] noise
        bool record_noise

        vector[vector[double]] x
        bool record_x

        vector[vector[double]] rprev
        bool record_rprev

        vector[vector[double]] r
        bool record_r

        vector[vector[double]] delta_x
        bool record_delta_x

        vector[vector[double]] x_mean
        bool record_x_mean

        # Targets
        vector[vector[double]] _sum_exc
        bool record__sum_exc

        vector[vector[double]] _sum_in
        bool record__sum_in

    # Projection 0 : Monitor
    cdef cppclass ProjRecorder0 (Monitor):
        ProjRecorder0(vector[int], int, int, long) except +

        vector[vector[vector[double]]] w
        bool record_w

    # Projection 1 : Monitor
    cdef cppclass ProjRecorder1 (Monitor):
        ProjRecorder1(vector[int], int, int, long) except +

        vector[double] effective_eta
        bool record_effective_eta

        vector[double] learning_phase
        bool record_learning_phase

        vector[double] error
        bool record_error

        vector[double] mean_error
        bool record_mean_error

        vector[double] mean_mean_error
        bool record_mean_mean_error

        vector[double] max_weight_change
        bool record_max_weight_change

        vector[vector[vector[double]]] trace
        bool record_trace

        vector[vector[vector[double]]] delta_w
        bool record_delta_w

        vector[vector[vector[double]]] w
        bool record_w


    # Instances

    PopStruct0 pop0
    PopStruct1 pop1

    ProjStruct0 proj0
    ProjStruct1 proj1

    # Methods
    void initialize(double, long)
    void init_rng_dist()
    void setSeed(long)
    void run(int nbSteps) nogil
    int run_until(int steps, vector[int] populations, bool or_and)
    void step()

    # Time
    long getTime()
    void setTime(long)

    # dt
    double getDt()
    void setDt(double dt_)


    # Number of threads
    void setNumberThreads(int)


# Population wrappers

# Wrapper for population 0 (pop0)
cdef class pop0_wrapper :

    def __cinit__(self, size, max_delay):

        pop0.set_size(size)
        pop0.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop0.get_size()
    # Reset the population
    def reset(self):
        pop0.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop0.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop0.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop0.set_active(val)


    # Local parameter r
    cpdef np.ndarray get_r(self):
        return np.array(pop0.get_r())
    cpdef set_r(self, np.ndarray value):
        pop0.set_r( value )
    cpdef double get_single_r(self, int rank):
        return pop0.get_single_r(rank)
    cpdef set_single_r(self, int rank, value):
        pop0.set_single_r(rank, value)


    # Targets





    # memory management
    def size_in_bytes(self):
        return pop0.size_in_bytes()

    def clear(self):
        return pop0.clear()

# Wrapper for population 1 (pop1)
cdef class pop1_wrapper :

    def __cinit__(self, size, max_delay):

        pop1.set_size(size)
        pop1.set_max_delay(max_delay)
    # Number of neurons
    property size:
        def __get__(self):
            return pop1.get_size()
    # Reset the population
    def reset(self):
        pop1.reset()
    # Set the maximum delay of outgoing projections
    def set_max_delay(self, val):
        pop1.set_max_delay(val)
    # Updates the maximum delay of outgoing projections and rebuilds the arrays
    def update_max_delay(self, val):
        pop1.update_max_delay(val)
    # Allows the population to compute
    def activate(self, bool val):
        pop1.set_active(val)


    # Global parameter tau
    cpdef double get_tau(self):
        return pop1.get_tau()
    cpdef set_tau(self, double value):
        pop1.set_tau(value)

    # Local parameter constant
    cpdef np.ndarray get_constant(self):
        return np.array(pop1.get_constant())
    cpdef set_constant(self, np.ndarray value):
        pop1.set_constant( value )
    cpdef double get_single_constant(self, int rank):
        return pop1.get_single_constant(rank)
    cpdef set_single_constant(self, int rank, value):
        pop1.set_single_constant(rank, value)

    # Global parameter alpha
    cpdef double get_alpha(self):
        return pop1.get_alpha()
    cpdef set_alpha(self, double value):
        pop1.set_alpha(value)

    # Global parameter f
    cpdef double get_f(self):
        return pop1.get_f()
    cpdef set_f(self, double value):
        pop1.set_f(value)

    # Global parameter A
    cpdef double get_A(self):
        return pop1.get_A()
    cpdef set_A(self, double value):
        pop1.set_A(value)

    # Local variable perturbation
    cpdef np.ndarray get_perturbation(self):
        return np.array(pop1.get_perturbation())
    cpdef set_perturbation(self, np.ndarray value):
        pop1.set_perturbation( value )
    cpdef double get_single_perturbation(self, int rank):
        return pop1.get_single_perturbation(rank)
    cpdef set_single_perturbation(self, int rank, value):
        pop1.set_single_perturbation(rank, value)

    # Local variable noise
    cpdef np.ndarray get_noise(self):
        return np.array(pop1.get_noise())
    cpdef set_noise(self, np.ndarray value):
        pop1.set_noise( value )
    cpdef double get_single_noise(self, int rank):
        return pop1.get_single_noise(rank)
    cpdef set_single_noise(self, int rank, value):
        pop1.set_single_noise(rank, value)

    # Local variable x
    cpdef np.ndarray get_x(self):
        return np.array(pop1.get_x())
    cpdef set_x(self, np.ndarray value):
        pop1.set_x( value )
    cpdef double get_single_x(self, int rank):
        return pop1.get_single_x(rank)
    cpdef set_single_x(self, int rank, value):
        pop1.set_single_x(rank, value)

    # Local variable rprev
    cpdef np.ndarray get_rprev(self):
        return np.array(pop1.get_rprev())
    cpdef set_rprev(self, np.ndarray value):
        pop1.set_rprev( value )
    cpdef double get_single_rprev(self, int rank):
        return pop1.get_single_rprev(rank)
    cpdef set_single_rprev(self, int rank, value):
        pop1.set_single_rprev(rank, value)

    # Local variable r
    cpdef np.ndarray get_r(self):
        return np.array(pop1.get_r())
    cpdef set_r(self, np.ndarray value):
        pop1.set_r( value )
    cpdef double get_single_r(self, int rank):
        return pop1.get_single_r(rank)
    cpdef set_single_r(self, int rank, value):
        pop1.set_single_r(rank, value)

    # Local variable delta_x
    cpdef np.ndarray get_delta_x(self):
        return np.array(pop1.get_delta_x())
    cpdef set_delta_x(self, np.ndarray value):
        pop1.set_delta_x( value )
    cpdef double get_single_delta_x(self, int rank):
        return pop1.get_single_delta_x(rank)
    cpdef set_single_delta_x(self, int rank, value):
        pop1.set_single_delta_x(rank, value)

    # Local variable x_mean
    cpdef np.ndarray get_x_mean(self):
        return np.array(pop1.get_x_mean())
    cpdef set_x_mean(self, np.ndarray value):
        pop1.set_x_mean( value )
    cpdef double get_single_x_mean(self, int rank):
        return pop1.get_single_x_mean(rank)
    cpdef set_single_x_mean(self, int rank, value):
        pop1.set_single_x_mean(rank, value)


    # Targets
    cpdef np.ndarray get_sum_exc(self):
        return np.array(pop1._sum_exc)
    cpdef np.ndarray get_sum_in(self):
        return np.array(pop1._sum_in)





    # memory management
    def size_in_bytes(self):
        return pop1.size_in_bytes()

    def clear(self):
        return pop1.clear()


# Projection wrappers

# Wrapper for projection 0
cdef class proj0_wrapper :

    def __cinit__(self, synapses):

        cdef LIL syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()
        proj0.set_size( size )
        proj0.set_post_rank( syn.post_rank )
        proj0.set_pre_rank( syn.pre_rank )

        proj0.set_w(syn.w)




    property size:
        def __get__(self):
            return proj0.get_size()

    def nb_synapses(self, int n):
        return proj0.nb_synapses(n)

    # Transmission flag
    def _get_transmission(self):
        return proj0._transmission
    def _set_transmission(self, bool l):
        proj0._transmission = l

    # Update flag
    def _get_update(self):
        return proj0._update
    def _set_update(self, bool l):
        proj0._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj0._plasticity
    def _set_plasticity(self, bool l):
        proj0._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj0._update_period
    def _set_update_period(self, int l):
        proj0._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj0._update_offset
    def _set_update_offset(self, long l):
        proj0._update_offset = l


    # Connectivity
    def post_rank(self):
        return proj0.get_post_rank()
    def set_post_rank(self, val):
        proj0.set_post_rank(val)
        proj0.inverse_connectivity_matrix()
    def pre_rank(self, int n):
        return proj0.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj0.get_pre_rank()
    def set_pre_rank(self, val):
        proj0.set_pre_rank(val)
        proj0.inverse_connectivity_matrix()

    # Local variable w
    def get_w(self):
        return proj0.get_w()
    def set_w(self, value):
        proj0.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj0.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, vector[double] value):
        proj0.set_dendrite_w(rank, value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj0.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj0.set_synapse_w(rank_post, rank_pre, value)







    # memory management
    def size_in_bytes(self):
        return proj0.size_in_bytes()

    def clear(self):
        return proj0.clear()

# Wrapper for projection 1
cdef class proj1_wrapper :

    def __cinit__(self, synapses):

        cdef LIL syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()
        proj1.set_size( size )
        proj1.set_post_rank( syn.post_rank )
        proj1.set_pre_rank( syn.pre_rank )

        proj1.set_w(syn.w)




    property size:
        def __get__(self):
            return proj1.get_size()

    def nb_synapses(self, int n):
        return proj1.nb_synapses(n)

    # Transmission flag
    def _get_transmission(self):
        return proj1._transmission
    def _set_transmission(self, bool l):
        proj1._transmission = l

    # Update flag
    def _get_update(self):
        return proj1._update
    def _set_update(self, bool l):
        proj1._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj1._plasticity
    def _set_plasticity(self, bool l):
        proj1._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj1._update_period
    def _set_update_period(self, int l):
        proj1._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj1._update_offset
    def _set_update_offset(self, long l):
        proj1._update_offset = l


    # Connectivity
    def post_rank(self):
        return proj1.get_post_rank()
    def set_post_rank(self, val):
        proj1.set_post_rank(val)
        proj1.inverse_connectivity_matrix()
    def pre_rank(self, int n):
        return proj1.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj1.get_pre_rank()
    def set_pre_rank(self, val):
        proj1.set_pre_rank(val)
        proj1.inverse_connectivity_matrix()

    # Local variable w
    def get_w(self):
        return proj1.get_w()
    def set_w(self, value):
        proj1.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj1.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, vector[double] value):
        proj1.set_dendrite_w(rank, value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj1.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj1.set_synapse_w(rank_post, rank_pre, value)



    # Global parameter effective_eta
    def get_effective_eta(self):
        return proj1.get_effective_eta()
    def set_effective_eta(self, value):
        proj1.set_effective_eta(value)

    # Global parameter learning_phase
    def get_learning_phase(self):
        return proj1.get_learning_phase()
    def set_learning_phase(self, value):
        proj1.set_learning_phase(value)

    # Global parameter error
    def get_error(self):
        return proj1.get_error()
    def set_error(self, value):
        proj1.set_error(value)

    # Global parameter mean_error
    def get_mean_error(self):
        return proj1.get_mean_error()
    def set_mean_error(self, value):
        proj1.set_mean_error(value)

    # Global parameter mean_mean_error
    def get_mean_mean_error(self):
        return proj1.get_mean_mean_error()
    def set_mean_mean_error(self, value):
        proj1.set_mean_mean_error(value)

    # Global parameter max_weight_change
    def get_max_weight_change(self):
        return proj1.get_max_weight_change()
    def set_max_weight_change(self, value):
        proj1.set_max_weight_change(value)

    # Local variable trace
    def get_trace(self):
        return proj1.get_trace()
    def set_trace(self, value):
        proj1.set_trace( value )
    def get_dendrite_trace(self, int rank):
        return proj1.get_dendrite_trace(rank)
    def set_dendrite_trace(self, int rank, vector[double] value):
        proj1.set_dendrite_trace(rank, value)
    def get_synapse_trace(self, int rank_post, int rank_pre):
        return proj1.get_synapse_trace(rank_post, rank_pre)
    def set_synapse_trace(self, int rank_post, int rank_pre, double value):
        proj1.set_synapse_trace(rank_post, rank_pre, value)

    # Local variable delta_w
    def get_delta_w(self):
        return proj1.get_delta_w()
    def set_delta_w(self, value):
        proj1.set_delta_w( value )
    def get_dendrite_delta_w(self, int rank):
        return proj1.get_dendrite_delta_w(rank)
    def set_dendrite_delta_w(self, int rank, vector[double] value):
        proj1.set_dendrite_delta_w(rank, value)
    def get_synapse_delta_w(self, int rank_post, int rank_pre):
        return proj1.get_synapse_delta_w(rank_post, rank_pre)
    def set_synapse_delta_w(self, int rank_post, int rank_pre, double value):
        proj1.set_synapse_delta_w(rank_post, rank_pre, value)





    # memory management
    def size_in_bytes(self):
        return proj1.size_in_bytes()

    def clear(self):
        return proj1.clear()


# Monitor wrappers
cdef class Monitor_wrapper:
    cdef Monitor *thisptr
    def __cinit__(self, list ranks, int period, int period_offset, long offset):
        pass
    property ranks:
        def __get__(self): return self.thisptr.ranks
        def __set__(self, val): self.thisptr.ranks = val
    property period:
        def __get__(self): return self.thisptr.period_
        def __set__(self, val): self.thisptr.period_ = val
    property offset:
        def __get__(self): return self.thisptr.offset_
        def __set__(self, val): self.thisptr.offset_ = val
    property period_offset:
        def __get__(self): return self.thisptr.period_offset_
        def __set__(self, val): self.thisptr.period_offset_ = val

def add_recorder(Monitor_wrapper recorder):
    addRecorder(recorder.thisptr)
def remove_recorder(Monitor_wrapper recorder):
    removeRecorder(recorder.thisptr)


# Population Monitor wrapper
cdef class PopRecorder0_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, period_offset, long offset):
        self.thisptr = new PopRecorder0(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (<PopRecorder0 *>self.thisptr).size_in_bytes()

    def clear(self):
        (<PopRecorder0 *>self.thisptr).clear()


    property r:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).r
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).r = val
    property record_r:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).record_r
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).record_r = val
    def clear_r(self):
        (<PopRecorder0 *>self.thisptr).r.clear()

    # Targets
# Population Monitor wrapper
cdef class PopRecorder1_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, period_offset, long offset):
        self.thisptr = new PopRecorder1(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (<PopRecorder1 *>self.thisptr).size_in_bytes()

    def clear(self):
        (<PopRecorder1 *>self.thisptr).clear()


    property tau:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).tau
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).tau = val
    property record_tau:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_tau
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_tau = val
    def clear_tau(self):
        (<PopRecorder1 *>self.thisptr).tau.clear()

    property constant:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).constant
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).constant = val
    property record_constant:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_constant
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_constant = val
    def clear_constant(self):
        (<PopRecorder1 *>self.thisptr).constant.clear()

    property alpha:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).alpha
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).alpha = val
    property record_alpha:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_alpha
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_alpha = val
    def clear_alpha(self):
        (<PopRecorder1 *>self.thisptr).alpha.clear()

    property f:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).f
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).f = val
    property record_f:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_f
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_f = val
    def clear_f(self):
        (<PopRecorder1 *>self.thisptr).f.clear()

    property A:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).A
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).A = val
    property record_A:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_A
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_A = val
    def clear_A(self):
        (<PopRecorder1 *>self.thisptr).A.clear()

    property perturbation:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).perturbation
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).perturbation = val
    property record_perturbation:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_perturbation
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_perturbation = val
    def clear_perturbation(self):
        (<PopRecorder1 *>self.thisptr).perturbation.clear()

    property noise:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).noise
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).noise = val
    property record_noise:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_noise
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_noise = val
    def clear_noise(self):
        (<PopRecorder1 *>self.thisptr).noise.clear()

    property x:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).x
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).x = val
    property record_x:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_x
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_x = val
    def clear_x(self):
        (<PopRecorder1 *>self.thisptr).x.clear()

    property rprev:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).rprev
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).rprev = val
    property record_rprev:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_rprev
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_rprev = val
    def clear_rprev(self):
        (<PopRecorder1 *>self.thisptr).rprev.clear()

    property r:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).r
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).r = val
    property record_r:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_r
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_r = val
    def clear_r(self):
        (<PopRecorder1 *>self.thisptr).r.clear()

    property delta_x:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).delta_x
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).delta_x = val
    property record_delta_x:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_delta_x
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_delta_x = val
    def clear_delta_x(self):
        (<PopRecorder1 *>self.thisptr).delta_x.clear()

    property x_mean:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).x_mean
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).x_mean = val
    property record_x_mean:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_x_mean
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_x_mean = val
    def clear_x_mean(self):
        (<PopRecorder1 *>self.thisptr).x_mean.clear()

    # Targets
    property _sum_exc:
        def __get__(self): return (<PopRecorder1 *>self.thisptr)._sum_exc
        def __set__(self, val): (<PopRecorder1 *>self.thisptr)._sum_exc = val
    property record__sum_exc:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record__sum_exc
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record__sum_exc = val
    def clear__sum_exc(self):
        (<PopRecorder1 *>self.thisptr)._sum_exc.clear()

    property _sum_in:
        def __get__(self): return (<PopRecorder1 *>self.thisptr)._sum_in
        def __set__(self, val): (<PopRecorder1 *>self.thisptr)._sum_in = val
    property record__sum_in:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record__sum_in
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record__sum_in = val
    def clear__sum_in(self):
        (<PopRecorder1 *>self.thisptr)._sum_in.clear()

# Projection Monitor wrapper
cdef class ProjRecorder0_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, int period_offset, long offset):
        self.thisptr = new ProjRecorder0(ranks, period, period_offset, offset)

    property w:
        def __get__(self): return (<ProjRecorder0 *>self.thisptr).w
        def __set__(self, val): (<ProjRecorder0 *>self.thisptr).w = val
    property record_w:
        def __get__(self): return (<ProjRecorder0 *>self.thisptr).record_w
        def __set__(self, val): (<ProjRecorder0 *>self.thisptr).record_w = val
    def clear_w(self):
        (<ProjRecorder0 *>self.thisptr).w.clear()

# Projection Monitor wrapper
cdef class ProjRecorder1_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, int period_offset, long offset):
        self.thisptr = new ProjRecorder1(ranks, period, period_offset, offset)

    property effective_eta:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).effective_eta
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).effective_eta = val
    property record_effective_eta:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_effective_eta
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_effective_eta = val
    def clear_effective_eta(self):
        (<ProjRecorder1 *>self.thisptr).effective_eta.clear()

    property learning_phase:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).learning_phase
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).learning_phase = val
    property record_learning_phase:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_learning_phase
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_learning_phase = val
    def clear_learning_phase(self):
        (<ProjRecorder1 *>self.thisptr).learning_phase.clear()

    property error:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).error
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).error = val
    property record_error:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_error
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_error = val
    def clear_error(self):
        (<ProjRecorder1 *>self.thisptr).error.clear()

    property mean_error:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).mean_error
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).mean_error = val
    property record_mean_error:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_mean_error
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_mean_error = val
    def clear_mean_error(self):
        (<ProjRecorder1 *>self.thisptr).mean_error.clear()

    property mean_mean_error:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).mean_mean_error
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).mean_mean_error = val
    property record_mean_mean_error:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_mean_mean_error
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_mean_mean_error = val
    def clear_mean_mean_error(self):
        (<ProjRecorder1 *>self.thisptr).mean_mean_error.clear()

    property max_weight_change:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).max_weight_change
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).max_weight_change = val
    property record_max_weight_change:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_max_weight_change
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_max_weight_change = val
    def clear_max_weight_change(self):
        (<ProjRecorder1 *>self.thisptr).max_weight_change.clear()

    property trace:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).trace
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).trace = val
    property record_trace:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_trace
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_trace = val
    def clear_trace(self):
        (<ProjRecorder1 *>self.thisptr).trace.clear()

    property delta_w:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).delta_w
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).delta_w = val
    property record_delta_w:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_delta_w
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_delta_w = val
    def clear_delta_w(self):
        (<ProjRecorder1 *>self.thisptr).delta_w.clear()

    property w:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).w
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).w = val
    property record_w:
        def __get__(self): return (<ProjRecorder1 *>self.thisptr).record_w
        def __set__(self, val): (<ProjRecorder1 *>self.thisptr).record_w = val
    def clear_w(self):
        (<ProjRecorder1 *>self.thisptr).w.clear()


# User-defined functions


# User-defined constants


# Initialize the network
def pyx_create(double dt, long seed):
    initialize(dt, seed)

def pyx_init_rng_dist():
    init_rng_dist()

# Simple progressbar on the command line
def progress(count, total, status=''):
    """
    Prints a progress bar on the command line.

    adapted from: https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    Modification: The original code set the '\r' at the end, so the bar disappears when finished.
    I moved it to the front, so the last status remains.
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()

# Simulation for the given number of steps
def pyx_run(int nb_steps, progress_bar):
    cdef int nb, rest
    cdef int batch = 1000
    if nb_steps < batch:
        with nogil:
            run(nb_steps)
    else:
        nb = int(nb_steps/batch)
        rest = nb_steps % batch
        for i in range(nb):
            with nogil:
                run(batch)
            PyErr_CheckSignals()
            if nb > 1 and progress_bar:
                progress(i+1, nb, 'simulate()')
        if rest > 0:
            run(rest)

        if (progress_bar):
            print('\n')

# Simulation for the given number of steps except if a criterion is reached
def pyx_run_until(int nb_steps, list populations, bool mode):
    cdef int nb
    nb = run_until(nb_steps, populations, mode)
    return nb

# Simulate for one step
def pyx_step():
    step()

# Access time
def set_time(t):
    setTime(t)
def get_time():
    return getTime()

# Access dt
def set_dt(double dt):
    setDt(dt)
def get_dt():
    return getDt()


# Set number of threads
def set_number_threads(int n):
    setNumberThreads(n)


# Set seed
def set_seed(long seed):
    setSeed(seed)
