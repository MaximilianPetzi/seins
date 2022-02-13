#pragma once
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "sparse_matrix.hpp"

#include "pop0.hpp"
#include "pop1.hpp"



extern PopStruct0 pop0;
extern PopStruct1 pop1;

extern std::vector<std::mt19937> rng;

/////////////////////////////////////////////////////////////////////////////
// proj0: pop0 -> pop1 with target in
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0 : LILMatrix<int> {
    ProjStruct0() : LILMatrix<int>( 400, 9) {
    }


    void init_from_lil( std::vector<int> &row_indices,
                        std::vector< std::vector<int> > &column_indices,
                        std::vector< std::vector<double> > &values,
                        std::vector< std::vector<int> > &delays) {
        static_cast<LILMatrix<int>*>(this)->init_matrix_from_lil(row_indices, column_indices);


        // Local parameter w
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





    // Local parameter w
    std::vector< std::vector<double > > w;




    // Method called to initialize the projection
    void init_projection() {
    #ifdef _DEBUG
        std::cout << "ProjStruct0::init_projection()" << std::endl;
    #endif

        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;






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
        std::cout << "    ProjStruct0::compute_psp()" << std::endl;
    #endif

        int nb_post; int rk_post; int rk_pre; double sum;

        if (_transmission && pop1._active){


            nb_post = post_rank.size();

            #pragma omp for private(sum)
            for(int i = 0; i < nb_post; i++) {
                sum = 0.0;
                for(int j = 0; j < pre_rank[i].size(); j++) {
                    sum += pop0.r[pre_rank[i][j]]*w[i][j] ;
                }
                pop1._sum_in[post_rank[i]] += sum;
            }

        } // active

    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse(int tid) {
    #ifdef _TRACE_SIMULATION_STEPS
        std::cout << "    ProjStruct0::update_synapse()" << std::endl;
    #endif


    }

    // Post-synaptic events
    void post_event() {


    }

    // Accessors for default attributes
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }

    // Variable/Parameter access methods

    std::vector<std::vector<double>> get_local_attribute_all(std::string name) {

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable_all<double>(w);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_all: " << name << " not found" << std::endl;
        return std::vector<std::vector<double>>();
    }

    std::vector<double> get_local_attribute_row(std::string name, int rk_post) {

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable_row<double>(w, rk_post);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute_row: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_local_attribute(std::string name, int rk_post, int rk_pre) {

        if ( name.compare("w") == 0 ) {
            return get_matrix_variable<double>(w, rk_post, rk_pre);
        }


        // should not happen
        std::cerr << "ProjStruct0::get_local_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_local_attribute_all(std::string name, std::vector<std::vector<double>> value) {

        if ( name.compare("w") == 0 ) {
            update_matrix_variable_all<double>(w, value);

        }

    }

    void set_local_attribute_row(std::string name, int rk_post, std::vector<double> value) {

        if ( name.compare("w") == 0 ) {
            update_matrix_variable_row<double>(w, rk_post, value);

        }

    }

    void set_local_attribute(std::string name, int rk_post, int rk_pre, double value) {

        if ( name.compare("w") == 0 ) {
            update_matrix_variable<double>(w, rk_post, rk_pre, value);

        }

    }

    std::vector<double> get_semiglobal_attribute_all(std::string name) {


        // should not happen
        std::cerr << "ProjStruct0::get_semiglobal_attribute_all: " << name << " not found" << std::endl;
        return std::vector<double>();
    }

    double get_semiglobal_attribute(std::string name, int rk_post) {


        // should not happen
        std::cerr << "ProjStruct0::get_semiglobal_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_semiglobal_attribute_all(std::string name, std::vector<double> value) {

    }

    void set_semiglobal_attribute(std::string name, int rk_post, double value) {

    }

    double get_global_attribute(std::string name) {


        // should not happen
        std::cerr << "ProjStruct0::get_global_attribute: " << name << " not found" << std::endl;
        return 0.0;
    }

    void set_global_attribute(std::string name, double value) {

    }


    // Access additional


    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;

        // connectivity
        size_in_bytes += static_cast<LILMatrix<int>*>(this)->size_in_bytes();
        // local parameter w
        size_in_bytes += sizeof(double) * w.capacity();
        for(auto it = w.begin(); it != w.end(); it++)
            size_in_bytes += (it->capacity()) * sizeof(double);

        return size_in_bytes;
    }

    // Structural plasticity



    void clear() {
    #ifdef _DEBUG
        std::cout << "PopStruct0::clear()" << std::endl;
    #endif

    }
};

