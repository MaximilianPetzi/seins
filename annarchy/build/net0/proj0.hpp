#pragma once

#include "pop0.hpp"
#include "pop1.hpp"



extern PopStruct0 pop0;
extern PopStruct1 pop1;


/////////////////////////////////////////////////////////////////////////////
// proj0: pop0 -> pop1 with target in
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0{
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









    // Method called to initialize the projection
    void init_projection() {
        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;





        // Inverse the connectivity matrix if spiking neurons
        inverse_connectivity_matrix();







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

            std::vector<double> _pre_r = pop0.r;
            nb_post = post_rank.size();
            #pragma omp parallel for private(sum) firstprivate(_pre_r, nb_post)
            for(int i = 0; i < nb_post; i++) {
                sum = 0.0;
                for(int j = 0; j < pre_rank[i].size(); j++) {
                    sum += _pre_r[pre_rank[i][j]]*w[i][j] ;
                }
                pop1._sum_in[post_rank[i]] += sum;
            }

        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {


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




    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // local parameter w
        size_in_bytes += sizeof(double) * w.capacity();
        for(auto it = w.begin(); it != w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        return size_in_bytes;
    }

    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::clear()" << std::endl;
    #endif
        // Variables

    }
};

