# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
from libcpp.string cimport string
from math import ceil
import numpy as np
import sys
cimport numpy as np
cimport cython

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport LILConnectivity as LIL

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

        # Connectivity
        void init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)





        # Local Attributes
        vector[vector[double]] get_local_attribute_all(string)
        vector[double] get_local_attribute_row(string, int)
        double get_local_attribute(string, int, int)
        void set_local_attribute_all(string, vector[vector[double]])
        void set_local_attribute_row(string, int, vector[double])
        void set_local_attribute(string, int, int, double)

        # Semiglobal Attributes
        vector[double] get_semiglobal_attribute_all(string)
        double get_semiglobal_attribute(string, int)
        void set_semiglobal_attribute_all(string, vector[double])
        void set_semiglobal_attribute(string, int, double)

        # Global Attributes
        double get_global_attribute(string)
        void set_global_attribute(string, double)





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

        # Connectivity
        void init_from_lil(vector[int], vector[vector[int]], vector[vector[double]], vector[vector[int]])
        # Access connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_ranks()
        vector[int] get_dendrite_pre_rank(int)





        # Local Attributes
        vector[vector[double]] get_local_attribute_all(string)
        vector[double] get_local_attribute_row(string, int)
        double get_local_attribute(string, int, int)
        void set_local_attribute_all(string, vector[vector[double]])
        void set_local_attribute_row(string, int, vector[double])
        void set_local_attribute(string, int, int, double)

        # Semiglobal Attributes
        vector[double] get_semiglobal_attribute_all(string)
        double get_semiglobal_attribute(string, int)
        void set_semiglobal_attribute_all(string, vector[double])
        void set_semiglobal_attribute(string, int, double)

        # Global Attributes
        double get_global_attribute(string)
        void set_global_attribute(string, double)





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
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder0* get_instance(int)
        long int size_in_bytes()
        void clear()

        vector[vector[double]] r
        bool record_r

        # Targets
    # Population 1 (pop1) : Monitor
    cdef cppclass PopRecorder1 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        PopRecorder1* get_instance(int)
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
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder0* get_instance(int)

        vector[vector[vector[double]]] w
        bool record_w

    # Projection 1 : Monitor
    cdef cppclass ProjRecorder1 (Monitor):
        @staticmethod
        int create_instance(vector[int], int, int, long)
        @staticmethod
        ProjRecorder1* get_instance(int)

        vector[double] eta
        bool record_eta

        vector[double] learning_phase
        bool record_learning_phase

        vector[double] error
        bool record_error

        vector[double] mean_error
        bool record_mean_error

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
    void initialize(double)
    void init_rng_dist()
    void setSeed(long, int, bool)
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
    void setNumberThreads(int, vector[int])


# Population wrappers

# Wrapper for population 0 (pop0)
@cython.auto_pickle(True)
cdef class pop0_wrapper :

    def __init__(self, size, max_delay):

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
@cython.auto_pickle(True)
cdef class pop1_wrapper :

    def __init__(self, size, max_delay):

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
@cython.auto_pickle(True)
cdef class proj0_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil(self, synapses):
        proj0.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)


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

    # Access connectivity

    def post_rank(self):
        return proj0.get_post_rank()
    def pre_rank_all(self):
        return proj0.get_pre_ranks()
    def pre_rank(self, int n):
        return proj0.get_dendrite_pre_rank(n)




    # Local Attribute
    def set_local_attribute_all(self, name, value):
        cpp_string = name.encode('utf-8')
        proj0.set_local_attribute_all(cpp_string, value)

    def set_local_attribute_row(self, name, rk_post, value):
        cpp_string = name.encode('utf-8')
        proj0.set_local_attribute_row(cpp_string, rk_post, value)

    def set_local_attribute(self, name, rk_post, rk_pre, value):
        cpp_string = name.encode('utf-8')
        proj0.set_local_attribute(cpp_string, rk_post, rk_pre, value)

    def get_local_attribute_all(self, name):
        cpp_string = name.encode('utf-8')
        return proj0.get_local_attribute_all(cpp_string)

    def get_local_attribute_row(self, name, rk_post):
        cpp_string = name.encode('utf-8')
        return proj0.get_local_attribute_row(cpp_string, rk_post)

    def get_local_attribute(self, name, rk_post, rk_pre):
        cpp_string = name.encode('utf-8')
        return proj0.get_local_attribute(cpp_string, rk_post, rk_pre)

    # Semiglobal Attributes
    def get_semiglobal_attribute_all(self, name):
        cpp_string = name.encode('utf-8')
        return proj0.get_semiglobal_attribute_all(cpp_string)

    def get_semiglobal_attribute(self, name, rk_post):
        cpp_string = name.encode('utf-8')
        return proj0.get_semiglobal_attribute(cpp_string, rk_post)

    def set_semiglobal_attribute_all(self, name, value):
        cpp_string = name.encode('utf-8')
        proj0.set_semiglobal_attribute_all(cpp_string, value)

    def set_semiglobal_attribute(self, name, rk_post, value):
        cpp_string = name.encode('utf-8')
        proj0.set_semiglobal_attribute(cpp_string, rk_post, value)

    # Global Attributes
    def get_global_attribute(self, name):
        cpp_string = name.encode('utf-8')
        return proj0.get_global_attribute(cpp_string)

    def set_global_attribute(self, name, value):
        cpp_string = name.encode('utf-8')
        proj0.set_global_attribute(cpp_string, value)





    # memory management
    def size_in_bytes(self):
        return proj0.size_in_bytes()

    def clear(self):
        return proj0.clear()

# Wrapper for projection 1
@cython.auto_pickle(True)
cdef class proj1_wrapper :

    def __init__(self, ):
                    pass


    def init_from_lil(self, synapses):
        proj1.init_from_lil(synapses.post_rank, synapses.pre_rank, synapses.w, synapses.delay)


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

    # Access connectivity

    def post_rank(self):
        return proj1.get_post_rank()
    def pre_rank_all(self):
        return proj1.get_pre_ranks()
    def pre_rank(self, int n):
        return proj1.get_dendrite_pre_rank(n)




    # Local Attribute
    def set_local_attribute_all(self, name, value):
        cpp_string = name.encode('utf-8')
        proj1.set_local_attribute_all(cpp_string, value)

    def set_local_attribute_row(self, name, rk_post, value):
        cpp_string = name.encode('utf-8')
        proj1.set_local_attribute_row(cpp_string, rk_post, value)

    def set_local_attribute(self, name, rk_post, rk_pre, value):
        cpp_string = name.encode('utf-8')
        proj1.set_local_attribute(cpp_string, rk_post, rk_pre, value)

    def get_local_attribute_all(self, name):
        cpp_string = name.encode('utf-8')
        return proj1.get_local_attribute_all(cpp_string)

    def get_local_attribute_row(self, name, rk_post):
        cpp_string = name.encode('utf-8')
        return proj1.get_local_attribute_row(cpp_string, rk_post)

    def get_local_attribute(self, name, rk_post, rk_pre):
        cpp_string = name.encode('utf-8')
        return proj1.get_local_attribute(cpp_string, rk_post, rk_pre)

    # Semiglobal Attributes
    def get_semiglobal_attribute_all(self, name):
        cpp_string = name.encode('utf-8')
        return proj1.get_semiglobal_attribute_all(cpp_string)

    def get_semiglobal_attribute(self, name, rk_post):
        cpp_string = name.encode('utf-8')
        return proj1.get_semiglobal_attribute(cpp_string, rk_post)

    def set_semiglobal_attribute_all(self, name, value):
        cpp_string = name.encode('utf-8')
        proj1.set_semiglobal_attribute_all(cpp_string, value)

    def set_semiglobal_attribute(self, name, rk_post, value):
        cpp_string = name.encode('utf-8')
        proj1.set_semiglobal_attribute(cpp_string, rk_post, value)

    # Global Attributes
    def get_global_attribute(self, name):
        cpp_string = name.encode('utf-8')
        return proj1.get_global_attribute(cpp_string)

    def set_global_attribute(self, name, value):
        cpp_string = name.encode('utf-8')
        proj1.set_global_attribute(cpp_string, value)





    # memory management
    def size_in_bytes(self):
        return proj1.size_in_bytes()

    def clear(self):
        return proj1.clear()


# Monitor wrappers

# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder0_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder0.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder0.get_instance(self.id)).size_in_bytes()

    property r:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder0.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder0.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder0.get_instance(self.id)).r.clear()

    # Targets
# Population Monitor wrapper
@cython.auto_pickle(True)
cdef class PopRecorder1_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, period_offset, long offset):
        self.id = PopRecorder1.create_instance(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (PopRecorder1.get_instance(self.id)).size_in_bytes()

    property tau:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).tau
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).tau = val
    property record_tau:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_tau
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_tau = val
    def clear_tau(self):
        (PopRecorder1.get_instance(self.id)).tau.clear()

    property constant:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).constant
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).constant = val
    property record_constant:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_constant
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_constant = val
    def clear_constant(self):
        (PopRecorder1.get_instance(self.id)).constant.clear()

    property alpha:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).alpha
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).alpha = val
    property record_alpha:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_alpha
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_alpha = val
    def clear_alpha(self):
        (PopRecorder1.get_instance(self.id)).alpha.clear()

    property f:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).f
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).f = val
    property record_f:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_f
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_f = val
    def clear_f(self):
        (PopRecorder1.get_instance(self.id)).f.clear()

    property A:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).A
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).A = val
    property record_A:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_A
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_A = val
    def clear_A(self):
        (PopRecorder1.get_instance(self.id)).A.clear()

    property perturbation:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).perturbation
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).perturbation = val
    property record_perturbation:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_perturbation
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_perturbation = val
    def clear_perturbation(self):
        (PopRecorder1.get_instance(self.id)).perturbation.clear()

    property noise:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).noise
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).noise = val
    property record_noise:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_noise
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_noise = val
    def clear_noise(self):
        (PopRecorder1.get_instance(self.id)).noise.clear()

    property x:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).x
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).x = val
    property record_x:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_x
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_x = val
    def clear_x(self):
        (PopRecorder1.get_instance(self.id)).x.clear()

    property rprev:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).rprev
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).rprev = val
    property record_rprev:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_rprev
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_rprev = val
    def clear_rprev(self):
        (PopRecorder1.get_instance(self.id)).rprev.clear()

    property r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).r = val
    property record_r:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_r
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_r = val
    def clear_r(self):
        (PopRecorder1.get_instance(self.id)).r.clear()

    property delta_x:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).delta_x
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).delta_x = val
    property record_delta_x:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_delta_x
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_delta_x = val
    def clear_delta_x(self):
        (PopRecorder1.get_instance(self.id)).delta_x.clear()

    property x_mean:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).x_mean
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).x_mean = val
    property record_x_mean:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record_x_mean
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record_x_mean = val
    def clear_x_mean(self):
        (PopRecorder1.get_instance(self.id)).x_mean.clear()

    # Targets
    property _sum_exc:
        def __get__(self): return (PopRecorder1.get_instance(self.id))._sum_exc
        def __set__(self, val): (PopRecorder1.get_instance(self.id))._sum_exc = val
    property record__sum_exc:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record__sum_exc
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record__sum_exc = val
    def clear__sum_exc(self):
        (PopRecorder1.get_instance(self.id))._sum_exc.clear()

    property _sum_in:
        def __get__(self): return (PopRecorder1.get_instance(self.id))._sum_in
        def __set__(self, val): (PopRecorder1.get_instance(self.id))._sum_in = val
    property record__sum_in:
        def __get__(self): return (PopRecorder1.get_instance(self.id)).record__sum_in
        def __set__(self, val): (PopRecorder1.get_instance(self.id)).record__sum_in = val
    def clear__sum_in(self):
        (PopRecorder1.get_instance(self.id))._sum_in.clear()

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder0_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder0.create_instance(ranks, period, period_offset, offset)

    property w:
        def __get__(self): return (ProjRecorder0.get_instance(self.id)).w
        def __set__(self, val): (ProjRecorder0.get_instance(self.id)).w = val
    property record_w:
        def __get__(self): return (ProjRecorder0.get_instance(self.id)).record_w
        def __set__(self, val): (ProjRecorder0.get_instance(self.id)).record_w = val
    def clear_w(self):
        (ProjRecorder0.get_instance(self.id)).w.clear()

# Projection Monitor wrapper
@cython.auto_pickle(True)
cdef class ProjRecorder1_wrapper:
    cdef int id
    def __init__(self, list ranks, int period, int period_offset, long offset):
        self.id = ProjRecorder1.create_instance(ranks, period, period_offset, offset)

    property eta:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).eta
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).eta = val
    property record_eta:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_eta
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_eta = val
    def clear_eta(self):
        (ProjRecorder1.get_instance(self.id)).eta.clear()

    property learning_phase:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).learning_phase
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).learning_phase = val
    property record_learning_phase:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_learning_phase
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_learning_phase = val
    def clear_learning_phase(self):
        (ProjRecorder1.get_instance(self.id)).learning_phase.clear()

    property error:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).error
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).error = val
    property record_error:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_error
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_error = val
    def clear_error(self):
        (ProjRecorder1.get_instance(self.id)).error.clear()

    property mean_error:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).mean_error
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).mean_error = val
    property record_mean_error:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_mean_error
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_mean_error = val
    def clear_mean_error(self):
        (ProjRecorder1.get_instance(self.id)).mean_error.clear()

    property max_weight_change:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).max_weight_change
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).max_weight_change = val
    property record_max_weight_change:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_max_weight_change
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_max_weight_change = val
    def clear_max_weight_change(self):
        (ProjRecorder1.get_instance(self.id)).max_weight_change.clear()

    property trace:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).trace
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).trace = val
    property record_trace:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_trace
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_trace = val
    def clear_trace(self):
        (ProjRecorder1.get_instance(self.id)).trace.clear()

    property delta_w:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).delta_w
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).delta_w = val
    property record_delta_w:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_delta_w
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_delta_w = val
    def clear_delta_w(self):
        (ProjRecorder1.get_instance(self.id)).delta_w.clear()

    property w:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).w
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).w = val
    property record_w:
        def __get__(self): return (ProjRecorder1.get_instance(self.id)).record_w
        def __set__(self, val): (ProjRecorder1.get_instance(self.id)).record_w = val
    def clear_w(self):
        (ProjRecorder1.get_instance(self.id)).w.clear()


# User-defined functions


# User-defined constants


# Initialize the network
def pyx_create(double dt):
    initialize(dt)

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
def set_number_threads(int n, core_list):
    setNumberThreads(n, core_list)


# Set seed
def set_seed(long seed, int num_sources, use_seed_seq):
    setSeed(seed, num_sources, use_seed_seq)
