/*
 *  ANNarchy-version: 4.7.0b
 */
#pragma once
#include "ANNarchy.h"
#include <random>


extern double dt;
extern long int t;
extern std::vector<std::mt19937> rng;


///////////////////////////////////////////////////////////////
// Main Structure for the population of id 1 (pop1)
///////////////////////////////////////////////////////////////
struct PopStruct1{

    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    int max_delay; // Maximum number of steps to store for delayed synaptic transmission

    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    int get_max_delay() { return max_delay; }
    void set_max_delay(int d) { max_delay  = d; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }

    // workload assignment
    std::vector<int> chunks_;


    // Neuron specific parameters and variables

    // Global parameter tau
    double  tau ;

    // Local parameter constant
    std::vector< double > constant;

    // Global parameter alpha
    double  alpha ;

    // Global parameter f
    double  f ;

    // Global parameter A
    double  A ;

    // Local variable perturbation
    std::vector< double > perturbation;

    // Local variable noise
    std::vector< double > noise;

    // Local variable x
    std::vector< double > x;

    // Local variable rprev
    std::vector< double > rprev;

    // Local variable r
    std::vector< double > r;

    // Local variable delta_x
    std::vector< double > delta_x;

    // Local variable x_mean
    std::vector< double > x_mean;

    // Targets

    std::vector<double> _sum_exc;
    std::vector<double> _sum_in;
    // Global operations

    // Random numbers
    std::vector<double> rand_0;
    std::vector<std::uniform_real_distribution< double >> dist_rand_0;
        std::vector<double> rand_1;
    std::vector<std::uniform_real_distribution< double >> dist_rand_1;





    // Access methods to the parameters and variables

    // Global parameter tau
    double get_tau() { return tau; }
    void set_tau(double val) { tau = val; }

    // Local parameter constant
    std::vector< double > get_constant() { return constant; }
    double get_single_constant(int rk) { return constant[rk]; }
    void set_constant(std::vector< double > val) { constant = val; }
    void set_single_constant(int rk, double val) { constant[rk] = val; }

    // Global parameter alpha
    double get_alpha() { return alpha; }
    void set_alpha(double val) { alpha = val; }

    // Global parameter f
    double get_f() { return f; }
    void set_f(double val) { f = val; }

    // Global parameter A
    double get_A() { return A; }
    void set_A(double val) { A = val; }

    // Local variable perturbation
    std::vector< double > get_perturbation() { return perturbation; }
    double get_single_perturbation(int rk) { return perturbation[rk]; }
    void set_perturbation(std::vector< double > val) { perturbation = val; }
    void set_single_perturbation(int rk, double val) { perturbation[rk] = val; }

    // Local variable noise
    std::vector< double > get_noise() { return noise; }
    double get_single_noise(int rk) { return noise[rk]; }
    void set_noise(std::vector< double > val) { noise = val; }
    void set_single_noise(int rk, double val) { noise[rk] = val; }

    // Local variable x
    std::vector< double > get_x() { return x; }
    double get_single_x(int rk) { return x[rk]; }
    void set_x(std::vector< double > val) { x = val; }
    void set_single_x(int rk, double val) { x[rk] = val; }

    // Local variable rprev
    std::vector< double > get_rprev() { return rprev; }
    double get_single_rprev(int rk) { return rprev[rk]; }
    void set_rprev(std::vector< double > val) { rprev = val; }
    void set_single_rprev(int rk, double val) { rprev[rk] = val; }

    // Local variable r
    std::vector< double > get_r() { return r; }
    double get_single_r(int rk) { return r[rk]; }
    void set_r(std::vector< double > val) { r = val; }
    void set_single_r(int rk, double val) { r[rk] = val; }

    // Local variable delta_x
    std::vector< double > get_delta_x() { return delta_x; }
    double get_single_delta_x(int rk) { return delta_x[rk]; }
    void set_delta_x(std::vector< double > val) { delta_x = val; }
    void set_single_delta_x(int rk, double val) { delta_x[rk] = val; }

    // Local variable x_mean
    std::vector< double > get_x_mean() { return x_mean; }
    double get_single_x_mean(int rk) { return x_mean[rk]; }
    void set_x_mean(std::vector< double > val) { x_mean = val; }
    void set_single_x_mean(int rk, double val) { x_mean[rk] = val; }



    // Method called to initialize the data structures
    void init_population() {
        _active = true;

        // Global parameter tau
        tau = 0.0;

        // Local parameter constant
        constant = std::vector<double>(size, 0.0);

        // Global parameter alpha
        alpha = 0.0;

        // Global parameter f
        f = 0.0;

        // Global parameter A
        A = 0.0;

        // Local variable perturbation
        perturbation = std::vector<double>(size, 0.0);

        // Local variable noise
        noise = std::vector<double>(size, 0.0);

        // Local variable x
        x = std::vector<double>(size, 0.0);

        // Local variable rprev
        rprev = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);

        // Local variable delta_x
        delta_x = std::vector<double>(size, 0.0);

        // Local variable x_mean
        x_mean = std::vector<double>(size, 0.0);

        rand_0 = std::vector<double>(size, 0.0);

        rand_1 = std::vector<double>(size, 0.0);

        // Post-synaptic potential
        _sum_exc = std::vector<double>(size, 0.0);
        // Post-synaptic potential
        _sum_in = std::vector<double>(size, 0.0);






        // distribute the work load across available threads
        int nt = omp_get_max_threads();
        chunks_ = std::vector<int>(nt+1);

        // try to avoid conflicts on one cache-line
        int elem_per_cacheline = 64 / sizeof(double);
        int chunk_size = static_cast<int>(ceil( static_cast<double>(size) / (static_cast<double>(nt*elem_per_cacheline))) * elem_per_cacheline );

        // initialize chunks
        for (int t = 0; t < nt; t++)
            chunks_[t] = t*chunk_size;
        chunks_[nt] = std::min(nt*chunk_size, size);
    #ifdef _DEBUG
        std::cout << "Population chunks: " << std::endl;
        std::cout << "nt " << nt << " -> " << chunk_size << " : ";
        for (auto it = chunks_.begin(); it != chunks_.end(); it++)
            std::cout << *it << " ";
        std::cout << std::endl;
    #endif

    }

    // Method called to reset the population
    void reset() {



    }

    // Init rng dist
    void init_rng_dist() {

        dist_rand_0 = std::vector< std::uniform_real_distribution< double > >(omp_get_max_threads());
        #pragma omp parallel
        {
            dist_rand_0[omp_get_thread_num()] = std::uniform_real_distribution< double >(0.0, 1.0);
        }

        dist_rand_1 = std::vector< std::uniform_real_distribution< double > >(omp_get_max_threads());
        #pragma omp parallel
        {
            dist_rand_1[omp_get_thread_num()] = std::uniform_real_distribution< double >(-1.0, 1.0);
        }

    }

    // Method to draw new random numbers
    void update_rng(int tid) {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct1::update_rng()" << std::endl;
#endif

        #pragma omp single
        {
            if (_active){

                for(int i = 0; i < size; i++) {

                rand_0[i] = dist_rand_0[0](rng[0]);

                rand_1[i] = dist_rand_1[0](rng[0]);

                }
            }
        }

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops(int tid, int nt) {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Method to dynamically change the size of the queue for delayed variables
    void update_max_delay(int value) {

    }

    // Main method to update neural variables
    void update(int tid) {
#ifdef _TRACE_SIMULATION_STEPS
    std::cout << "    PopStruct1::update()" << std::endl;
#endif

        if( _active ) {

            // Updating the local variables
            #pragma omp simd
            for (int i = chunks_[tid]; i < chunks_[tid+1]; i++) {

                // perturbation = if Uniform(0.0, 1.0) < f/1000.: 1.0 else: 0.0
                perturbation[i] = (rand_0[i] < f/1000.0 ? 1.0 : 0.0);


                // noise = if perturbation > 0.5: A*Uniform(-1.0, 1.0) else: 0.0
                noise[i] = (perturbation[i] > 0.5 ? A*rand_1[i] : 0.0);


                // x += dt*(sum(in) + sum(exc) - x + noise)/tau
                x[i] += dt*(_sum_exc[i] + _sum_in[i] + noise[i] - x[i])/tau;


                // rprev = r
                rprev[i] = r[i];


                // r = if constant == 0.0: tanh(x) else: tanh(constant)
                r[i] = (constant[i] == 0.0 ? tanh(x[i]) : tanh(constant[i]));


                // delta_x = x - x_mean
                delta_x[i] = x[i] - x_mean[i];


                // x_mean = alpha * x_mean + (1 - alpha) * x
                x_mean[i] = alpha*x_mean[i] + x[i]*(1 - alpha);


            }

        } // active

    }

    void spike_gather(int tid, int num_threads) {

    }



    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(double);	// tau
        size_in_bytes += sizeof(double) * constant.capacity();	// constant
        size_in_bytes += sizeof(double);	// alpha
        size_in_bytes += sizeof(double);	// f
        size_in_bytes += sizeof(double);	// A
        // Variables
        size_in_bytes += sizeof(double) * perturbation.capacity();	// perturbation
        size_in_bytes += sizeof(double) * noise.capacity();	// noise
        size_in_bytes += sizeof(double) * x.capacity();	// x
        size_in_bytes += sizeof(double) * rprev.capacity();	// rprev
        size_in_bytes += sizeof(double) * r.capacity();	// r
        size_in_bytes += sizeof(double) * delta_x.capacity();	// delta_x
        size_in_bytes += sizeof(double) * x_mean.capacity();	// x_mean

        return size_in_bytes;
    }

    // Memory management: track the memory consumption
    void clear() {
        // Variables
        perturbation.clear();
        perturbation.shrink_to_fit();
        noise.clear();
        noise.shrink_to_fit();
        x.clear();
        x.shrink_to_fit();
        rprev.clear();
        rprev.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();
        delta_x.clear();
        delta_x.shrink_to_fit();
        x_mean.clear();
        x_mean.shrink_to_fit();

    }
};

