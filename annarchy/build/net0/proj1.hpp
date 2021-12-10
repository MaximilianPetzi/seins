#pragma once

#include "pop1.hpp"
#include "pop1.hpp"



extern PopStruct1 pop1;
extern PopStruct1 pop1;


/////////////////////////////////////////////////////////////////////////////
// proj1: pop1 -> pop1 with target exc
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct1{
    // Number of dendrites
    int size;

    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;


    // Connectivity
    std::vector<int> post_rank;
    std::vector< std::vector< int > > pre_rank;

    // LIL weights
    std::vector< std::vector< double > > w;






    // Global parameter effective_eta
    double  effective_eta ;

    // Global parameter learning_phase
    double  learning_phase ;

    // Global parameter error
    double  error ;

    // Global parameter mean_error
    double  mean_error ;

    // Global parameter mean_mean_error
    double  mean_mean_error ;

    // Global parameter max_weight_change
    double  max_weight_change ;

    // Local variable trace
    std::vector< std::vector<double > > trace;

    // Local variable delta_w
    std::vector< std::vector<double > > delta_w;




    // Method called to initialize the projection
    void init_projection() {
        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;





        // Inverse the connectivity matrix if spiking neurons
        inverse_connectivity_matrix();



        // Global parameter effective_eta
        effective_eta = 0.0;

        // Global parameter learning_phase
        learning_phase = 0.0;

        // Global parameter error
        error = 0.0;

        // Global parameter mean_error
        mean_error = 0.0;

        // Global parameter mean_mean_error
        mean_mean_error = 0.0;

        // Global parameter max_weight_change
        max_weight_change = 0.0;

        // Local variable trace
        trace = std::vector< std::vector<double> >(post_rank.size(), std::vector<double>());

        // Local variable delta_w
        delta_w = std::vector< std::vector<double> >(post_rank.size(), std::vector<double>());





    }

    // Spiking networks: inverse the connectivity matrix
    void inverse_connectivity_matrix() {

    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {

    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {

        int nb_post; double sum;

        if (_transmission && pop1._active){

            std::vector<double> _pre_r = pop1.r;std::vector<double> _pre_rprev = pop1.rprev;
            nb_post = post_rank.size();
            #pragma omp parallel for private(sum) firstprivate(_pre_r, _pre_rprev, nb_post)
            for(int i = 0; i < nb_post; i++) {
                sum = 0.0;
                for(int j = 0; j < pre_rank[i].size(); j++) {
                    sum += _pre_r[pre_rank[i][j]]*w[i][j] ;
                }
                pop1._sum_exc[post_rank[i]] += sum;
            }

        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {

        int rk_post, rk_pre;
        double _dt = dt * _update_period;

        // Check periodicity
        if(_transmission && _update && pop1._active && ( (t - _update_offset)%_update_period == 0L) ){
            // Global variables

            // Local variables
            #pragma omp parallel for private(rk_pre, rk_post) schedule(dynamic)
            for(int i = 0; i < post_rank.size(); i++){
                rk_post = post_rank[i]; // Get postsynaptic rank
                // Semi-global variables

                // Local variables
                for(int j = 0; j < pre_rank[i].size(); j++){
                    rk_pre = pre_rank[i][j]; // Get presynaptic rank

                    // trace += if learning_phase < 0.5: power(pre.rprev * (post.delta_x), 3) else: 0.0
                    trace[i][j] += (learning_phase < 0.5 ? power(pop1.delta_x[rk_post]*pop1.rprev[rk_pre], 3) : 0.0);


                    // delta_w = if learning_phase > 0.5: effective_eta * trace * (mean_error) * (error - mean_error) else: 0.0
                    delta_w[i][j] = (learning_phase > 0.5 ? effective_eta*mean_error*trace[i][j]*(error - mean_error) : 0.0);
                    if(delta_w[i][j] < -max_weight_change)
                        delta_w[i][j] = -max_weight_change;
                    if(delta_w[i][j] > max_weight_change)
                        delta_w[i][j] = max_weight_change;


                    // w -= if learning_phase > 0.5: delta_w else: 0.0
                    if(_plasticity){
                    w[i][j] -= (learning_phase > 0.5 ? delta_w[i][j] : 0.0);

                    }

                }
            }
        }

    }

    // Post-synaptic events
    void post_event() {


    }

    // Accessors for default attributes
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }

    // Additional access methods

    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }

    // Local parameter w
    std::vector<std::vector< double > > get_w() {
        std::vector< std::vector< double > > w_new(w.size(), std::vector<double>());
        for(int i = 0; i < w.size(); i++) {
            w_new[i] = std::vector<double>(w[i].begin(), w[i].end());
        }
        return w_new;
    }
    std::vector< double > get_dendrite_w(int rk) { return std::vector<double>(w[rk].begin(), w[rk].end()); }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) {
        w = std::vector< std::vector<double> >( value.size(), std::vector<double>() );
        for(int i = 0; i < value.size(); i++) {
            w[i] = std::vector<double>(value[i].begin(), value[i].end());
        }
    }
    void set_dendrite_w(int rk, std::vector< double > value) { w[rk] = std::vector<double>(value.begin(), value.end()); }
    void set_synapse_w(int rk_post, int rk_pre, double value) { w[rk_post][rk_pre] = value; }


    // Global parameter effective_eta
    double get_effective_eta() { return effective_eta; }
    void set_effective_eta(double value) { effective_eta = value; }

    // Global parameter learning_phase
    double get_learning_phase() { return learning_phase; }
    void set_learning_phase(double value) { learning_phase = value; }

    // Global parameter error
    double get_error() { return error; }
    void set_error(double value) { error = value; }

    // Global parameter mean_error
    double get_mean_error() { return mean_error; }
    void set_mean_error(double value) { mean_error = value; }

    // Global parameter mean_mean_error
    double get_mean_mean_error() { return mean_mean_error; }
    void set_mean_mean_error(double value) { mean_mean_error = value; }

    // Global parameter max_weight_change
    double get_max_weight_change() { return max_weight_change; }
    void set_max_weight_change(double value) { max_weight_change = value; }

    // Local variable trace
    std::vector<std::vector< double > > get_trace() { return trace; }
    std::vector<double> get_dendrite_trace(int rk) { return trace[rk]; }
    double get_synapse_trace(int rk_post, int rk_pre) { return trace[rk_post][rk_pre]; }
    void set_trace(std::vector<std::vector< double > >value) { trace = value; }
    void set_dendrite_trace(int rk, std::vector<double> value) { trace[rk] = value; }
    void set_synapse_trace(int rk_post, int rk_pre, double value) { trace[rk_post][rk_pre] = value; }

    // Local variable delta_w
    std::vector<std::vector< double > > get_delta_w() { return delta_w; }
    std::vector<double> get_dendrite_delta_w(int rk) { return delta_w[rk]; }
    double get_synapse_delta_w(int rk_post, int rk_pre) { return delta_w[rk_post][rk_pre]; }
    void set_delta_w(std::vector<std::vector< double > >value) { delta_w = value; }
    void set_dendrite_delta_w(int rk, std::vector<double> value) { delta_w[rk] = value; }
    void set_synapse_delta_w(int rk_post, int rk_pre, double value) { delta_w[rk_post][rk_pre] = value; }



    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // local variable trace
        size_in_bytes += sizeof(double) * trace.capacity();
        for(auto it = trace.begin(); it != trace.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // local variable delta_w
        size_in_bytes += sizeof(double) * delta_w.capacity();
        for(auto it = delta_w.begin(); it != delta_w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // local variable w
        size_in_bytes += sizeof(double) * w.capacity();
        for(auto it = w.begin(); it != w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // global parameter effective_eta
        size_in_bytes += sizeof(double);	// effective_eta
        // global parameter learning_phase
        size_in_bytes += sizeof(double);	// learning_phase
        // global parameter error
        size_in_bytes += sizeof(double);	// error
        // global parameter mean_error
        size_in_bytes += sizeof(double);	// mean_error
        // global parameter mean_mean_error
        size_in_bytes += sizeof(double);	// mean_mean_error
        // global parameter max_weight_change
        size_in_bytes += sizeof(double);	// max_weight_change

        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct1::clear()" << std::endl;
    #endif
        // Variables
        trace.clear();
        trace.shrink_to_fit();
        delta_w.clear();
        delta_w.shrink_to_fit();
        w.clear();
        w.shrink_to_fit();

    }
};

