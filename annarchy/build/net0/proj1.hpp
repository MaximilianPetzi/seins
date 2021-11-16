#pragma once
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "sparse_matrix.hpp"

#include "pop1.hpp"
#include "pop1.hpp"



extern PopStruct1 pop1;
extern PopStruct1 pop1;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj1: pop1 -> pop1 with target exc
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct1 : LILMatrix<int> {
    ProjStruct1() : LILMatrix<int>( 800, 800) {
    }


    void init_from_lil( std::vector<int> &row_indices,
                        std::vector< std::vector<int> > &column_indices,
                        std::vector< std::vector<double> > &values,
                        std::vector< std::vector<int> > &delays) {
        static_cast<LILMatrix<int>*>(this)->init_matrix_from_lil(row_indices, column_indices);


        // Local variable w
        w = init_matrix_variable<double>(static_cast<double>(0.0));
        update_matrix_variable_all<double>(w, values);


    #ifdef _DEBUG_CONN
        static_cast<LILMatrix<int>*>(this)->print_data_representation();
    #endif
    }





    // Number of dendrites
    int size;

    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;





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

    // Local parameter trace
    std::vector< std::vector<double > > trace;

    // Global parameter eta_lr
    double  eta_lr ;

    // Global parameter eta
    double  eta ;

    // Global parameter effective_eta
    double  effective_eta ;

    // Local parameter delta_w
    std::vector< std::vector<double > > delta_w;

    // Local parameter w
    std::vector< std::vector<double > > w;




    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct1::init_projection()" << std::endl;
    #endif

        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;



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
        trace = init_matrix_variable<double>(static_cast<double>(0.0));

        // Global variable eta_lr
        eta_lr = 0.0;

        // Global variable eta
        eta = 0.0;

        // Global variable effective_eta
        effective_eta = 0.0;

        // Local variable delta_w
        delta_w = init_matrix_variable<double>(static_cast<double>(0.0));




    }

    // Spiking networks: reset the ring buffer when non-uniform
    void reset_ring_buffer() {

    }

    // Spiking networks: update maximum delay when non-uniform
    void update_max_delay(int d){

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct1::compute_psp()" << std::endl;
    #endif

        int nb_post; int rk_post; int rk_pre; double sum;

        if (_transmission && pop1._active){


            nb_post = post_rank.size();

            #pragma omp for private(sum)
            for(int i = 0; i < nb_post; i++) {
                sum = 0.0;
                for(int j = 0; j < pre_rank[i].size(); j++) {
                    sum += pop1.r[pre_rank[i][j]]*w[i][j] ;
                }
                pop1._sum_exc[post_rank[i]] += sum;
            }

        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse(int tid) {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct1::update_synapse()" << std::endl;
    #endif

        int rk_post, rk_pre;
        double _dt = dt * _update_period;

        // Check periodicity
        if(_transmission && _update && pop1._active && ( (t - _update_offset)%_update_period == 0L) ){
            // Global variables

                // eta_lr=0.0
                eta_lr = 0.0;


                // eta += if learning_phase > 0.5: -eta_lr*(mean_error-mean_mean_error) else: 0.0
                eta += (learning_phase > 0.5 ? (-eta_lr)*(mean_error - mean_mean_error) : 0.0);


                // effective_eta = if learning_phase > 0.5: eta else: 0.0
                effective_eta = (learning_phase > 0.5 ? eta : 0.0);


            // Local variables

            #pragma omp for private(rk_post, rk_pre) firstprivate(dt)
            for(int i = 0; i < post_rank.size(); i++){
                rk_post = post_rank[i]; // Get postsynaptic rank
                // Semi-global variables

                // Local variables
                for(int j = 0; j < pre_rank[i].size(); j++){
                    rk_pre = pre_rank[i][j]; // Get presynaptic rank

                    // trace += if learning_phase < 0.5: power(pre.rprev * (post.delta_x), 3) else: 0.0
                    trace[i][j] += (learning_phase < 0.5 ? power(pop1.delta_x[post_rank[i]]*pop1.rprev[pre_rank[i][j]], 3) : 0.0);


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

    // Variable/Parameter access methods

    std::vector<std::vector<double>> get_local_attribute_all(std::string name) {

        if ( name.compare("trace") == 0 ) {
            return get_matrix_variable_all<double>(trace);
        }

        if ( name.compare("delta_w") == 0 ) {
            return get_matrix_variable_all<double>(delta_w);
        }

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable_all<double>(w);
        }


        // should not happen
        std::cerr << "ProjStruct1::get_local_attribute_all: " << name << " not found" << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<double> get_local_attribute_row(std::string name, int rk_post) {

        if ( name.compare("trace") == 0 ) {
            return get_matrix_variable_row<double>(trace, rk_post);
        }

        if ( name.compare("delta_w") == 0 ) {
            return get_matrix_variable_row<double>(delta_w, rk_post);
        }

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable_row<double>(w, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct1::get_local_attribute_row: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute(std::string name, int rk_post, int rk_pre) {

        if ( name.compare("trace") == 0 ) {
            return get_matrix_variable<double>(trace, rk_post, rk_pre);
        }

        if ( name.compare("delta_w") == 0 ) {
            return get_matrix_variable<double>(delta_w, rk_post, rk_pre);
        }

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable<double>(w, rk_post, rk_pre);
        }


        // should not happen
        std::cerr << "ProjStruct1::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all(std::string name, std::vector<std::vector<double>> value) {

        if ( name.compare("trace") == 0 ) {
            update_matrix_variable_all<double>(trace, value);

        }

        if ( name.compare("delta_w") == 0 ) {
            update_matrix_variable_all<double>(delta_w, value);

        }

        if ( name.compare("w") == 0 ) {
            update_matrix_variable_all<double>(w, value);

        }

    }

    void set_local_attribute_row(std::string name, int rk_post, std::vector<double> value) {

        if ( name.compare("trace") == 0 ) {
            update_matrix_variable_row<double>(trace, rk_post, value);

        }

        if ( name.compare("delta_w") == 0 ) {
            update_matrix_variable_row<double>(delta_w, rk_post, value);

        }

        if ( name.compare("w") == 0 ) {
            update_matrix_variable_row<double>(w, rk_post, value);

        }

    }

    void set_local_attribute(std::string name, int rk_post, int rk_pre, double value) {

        if ( name.compare("trace") == 0 ) {
            update_matrix_variable<double>(trace, rk_post, rk_pre, value);

        }

        if ( name.compare("delta_w") == 0 ) {
            update_matrix_variable<double>(delta_w, rk_post, rk_pre, value);

        }

        if ( name.compare("w") == 0 ) {
            update_matrix_variable<double>(w, rk_post, rk_pre, value);

        }

    }

    std::vector<double> get_semiglobal_attribute_all(std::string name) {


        // should not happen
        std::cerr << "ProjStruct1::get_semiglobal_attribute_all: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_semiglobal_attribute(std::string name, int rk_post) {


        // should not happen
        std::cerr << "ProjStruct1::get_semiglobal_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_semiglobal_attribute_all(std::string name, std::vector<double> value) {

    }

    void set_semiglobal_attribute(std::string name, int rk_post, double value) {

    }

    double get_global_attribute(std::string name) {

        if ( name.compare("learning_phase") == 0 ) {
            return learning_phase;
        }

        if ( name.compare("error") == 0 ) {
            return error;
        }

        if ( name.compare("mean_error") == 0 ) {
            return mean_error;
        }

        if ( name.compare("mean_mean_error") == 0 ) {
            return mean_mean_error;
        }

        if ( name.compare("max_weight_change") == 0 ) {
            return max_weight_change;
        }

        if ( name.compare("eta_lr") == 0 ) {
            return eta_lr;
        }

        if ( name.compare("eta") == 0 ) {
            return eta;
        }

        if ( name.compare("effective_eta") == 0 ) {
            return effective_eta;
        }


        // should not happen
        std::cerr << "ProjStruct1::get_global_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute(std::string name, double value) {

        if ( name.compare("learning_phase") == 0 ) {
            learning_phase = value;

        }

        if ( name.compare("error") == 0 ) {
            error = value;

        }

        if ( name.compare("mean_error") == 0 ) {
            mean_error = value;

        }

        if ( name.compare("mean_mean_error") == 0 ) {
            mean_mean_error = value;

        }

        if ( name.compare("max_weight_change") == 0 ) {
            max_weight_change = value;

        }

        if ( name.compare("eta_lr") == 0 ) {
            eta_lr = value;

        }

        if ( name.compare("eta") == 0 ) {
            eta = value;

        }

        if ( name.compare("effective_eta") == 0 ) {
            effective_eta = value;

        }

    }


    // Access additional


    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<LILMatrix<int>*>(this)->size_in_bytes();
        // local variable trace
        size_in_bytes += sizeof(double) * trace.capacity();
        for(auto it = trace.begin(); it != trace.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // global variable eta_lr
        size_in_bytes += sizeof(double);	// eta_lr
        // global variable eta
        size_in_bytes += sizeof(double);	// eta
        // global variable effective_eta
        size_in_bytes += sizeof(double);	// effective_eta
        // local variable delta_w
        size_in_bytes += sizeof(double) * delta_w.capacity();
        for(auto it = delta_w.begin(); it != delta_w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
        // local variable w
        size_in_bytes += sizeof(double) * w.capacity();
        for(auto it = w.begin(); it != w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);
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

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct1::clear()" << std::endl;
    #endif

    }
};

