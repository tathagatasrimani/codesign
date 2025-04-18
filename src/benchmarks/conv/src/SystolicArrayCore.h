#ifndef SYSTOLIC_ARRAY_CORE_H
#define SYSTOLIC_ARRAY_CORE_H

#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>

#include "ProcessingElement.h"
#include "Fifo.h"

// Define this macro for debug logging
#define HLS_DEBUG 0
#if HLS_DEBUG
#ifndef __SYNTHESIS__
#include <iostream>
#include <fstream>
#include <string>

// Only works for square arrays
template <typename T>
void log_matrix(std::ofstream& file, T* data, int iteration, int side_length) {
    file << "Iteration: " << iteration << '\n';
    for (int r = 0; r < side_length; r++) {
        for (int c = 0; c < side_length; c++) {
            file << int(data[r][c].to_int()) << ' ';
        }
        file << '\n';
    }
    file << '\n';
}
#endif
#endif


struct LoopIndices{
    uint_16 ic1_idx;
    uint_16 fx_idx;
    uint_16 fy_idx;
};



template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int OC0, int IC0>
class SystolicArrayCore
{
    #if HLS_DEBUG
    #ifndef __SYNTHESIS__
    // Create log file information
    std::ofstream input_file;
    std::ofstream weight_file;
    std::ofstream psum_file;
    #endif
    #endif


public:
    SystolicArrayCore() {
        #if HLS_DEBUG
        #ifndef __SYNTHESIS__

        // Creates filenames
        std::string input_filename = "input_file";
        std::string weight_filename = "weight_file";
        std::string psum_filename = "psum_file";

        // Opens log files when debugging
        input_file.open(input_filename.c_str());
        weight_file.open(weight_filename.c_str());
        psum_file.open(psum_filename.c_str());
        bool open_success = true;
        open_success = open_success && input_file.is_open();
        open_success = open_success && weight_file.is_open();
        open_success = open_success && psum_file.is_open();

        if (!open_success) {
            std::cerr << "Failed to open one or more log files." << std::endl;
        }
        #endif
        #endif
    }

#pragma hls_design interface
#pragma hls_pipeline_init_interval 1
    void CCS_BLOCK(run)(
        ac_channel<PackedInt<INPUT_PRECISION, IC0> > &input, 
        ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &weight, 
        ac_channel<PackedInt<OUTPUT_PRECISION, OC0> > &output,
        ac_channel<Params> &paramsIn,
        ac_channel<LoopIndices> &loopIndicesIn)
    {
        #ifndef __SYNTHESIS__
        // assert(params.OX0 * params.OY0 <= ACCUMULATION_BUFFER_SIZE);
        // Debug example:
        // printf("paramsIn channel size: %d\n", paramsIn.size());
        // printf("loopIndicesIn channel size: %d\n", loopIndicesIn.size());
        // printf("weight channel size: %d\n", weight.size());
        // printf("input channel size: %d\n\n", input.size());
        #endif

        #ifndef __SYNTHESIS__
        while(loopIndicesIn.available(1))
        #endif
        {
            // -------------------------------
            // Read in the params and loop indices from the channel
            // Your code starts here
            // -------------------------------
            Params params = paramsIn.read();
            LoopIndices loopIndices = loopIndicesIn.read();
            // -------------------------------
            // Your code ends here
            // -------------------------------


            // -------------------------------
            // Create a loop for a "run" of the systolic array.
            // The number of steps in a run of the systolic array is equal to:
            // the ramp-up time + number of pixels + flush time
            // Your code starts here
            // -------------------------------
            uint_16 step_bound = OC0+IC0+(params.OX0*params.OY0)-1;
            LABEL(INNER_LOOP) for (uint_16 step = 0; step < OC0_MAX + IC0_MAX + OX0_MAX * OY0_MAX - 1; ++step) { // loop inside each image tile
            // -------------------------------
            // Your code ends here 
            // You should now be in the body of the loop
            // -------------------------------

                // -------------------------------
                // If you are in the ramp up time, read in weights from the channel
                // and store it in the weights array
                // Your code starts here
                // -------------------------------
                if (step < IC0) {       
                    PackedInt<WEIGHT_PRECISION, OC0> w_row = weight.read();
                    #pragma hls_unroll yes
                    for(int j = 0; j < OC0; j++){
                            weight_reg[step][j] = w_row.value[j];
                    }
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------
                
                
                PackedInt<INPUT_PRECISION, IC0> in_col;

                // -------------------------------
                // Read inputs from the channel and store in the variable in_col
                // Note: you don't read in any inputs during the flush time
                // Your code starts here
                // -------------------------------
                if (step < (params.OX0*params.OY0)) {        
                    in_col = input.read();
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------

                // Debug example:        
                // printf("in_col: %s\n", in_col.to_string().c_str());


                /*
                 * FIFOs for inputs coming in to the systolic array
                 * assign values to in_col, and the skewed version will be in input_buf
                 */
                PackedInt<INPUT_PRECISION, IC0> input_buf;

                #define INPUT_FIFO_BODY(z,i,unused) \
                    IDTYPE BOOST_PP_CAT(input_fifo_output_, i); \
                    IDTYPE BOOST_PP_CAT(input_fifo_input_, i) = in_col.value[i]; \
                    BOOST_PP_CAT(input_fifo_, i).run( BOOST_PP_CAT(input_fifo_input_, i) , BOOST_PP_CAT(input_fifo_output_, i) ); \
                    input_buf.value[i] = BOOST_PP_CAT(input_fifo_output_, i);
                
                REPEAT(INPUT_FIFO_BODY)

                // -------------------------------
                // Assign values from input_buf into the registers for the first column of PEs
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                LABEL(INIT_IN) for(int i = 0; i < IC0; ++i) {
                    input_reg[i][0] = input_buf.value[i];
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------

                PackedInt<OUTPUT_PRECISION, OC0> psum_buf;
                
                // -------------------------------
                // Set partial outputs for the array to psum_buf.
                // Depending on the loop index, the partial output will be 0 or a value from the accumulation buffer
                // Your code starts here
                // -------------------------------
                if(step < (params.OX0*params.OY0)){
                    // initial partial output of 0
                    if(loopIndices.ic1_idx == 0 && loopIndices.fx_idx == 0 && loopIndices.fy_idx == 0) {
                        #pragma hls_unroll yes
                        for(int j = 0; j < OC0; j++){
                            psum_buf.value[j].template set_val<AC_VAL_0>();
                        }
                    }
                    else{ // read partial output from accumulation buffer
                        #pragma hls_unroll yes
                        for(int j = 0; j < OC0; j++){
                            psum_buf.value[j] = accumulation_buffer[step][j];
                        }
                    }
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------
                
                // Debug example:
                // printf("psum_buf: %s\n", psum_buf.to_string().c_str());

                /*
                 * FIFOs for partial outputs coming in to the systolic array
                 * assign values to psum_buf, and the skewed version will be in output_buf
                 */
                PackedInt<OUTPUT_PRECISION, OC0> output_buf;
                #define ACCUM_FIFO_BODY(z,i,unused) \
                    ODTYPE BOOST_PP_CAT(psum_fifo_output_, i); \
                    ODTYPE BOOST_PP_CAT(psum_fifo_input_, i) = psum_buf.value[i]; \
                    BOOST_PP_CAT(psum_fifo_, i).run( BOOST_PP_CAT(psum_fifo_input_, i) , BOOST_PP_CAT(psum_fifo_output_, i) ); \
                    output_buf.value[i] = BOOST_PP_CAT(psum_fifo_output_, i);
                
                REPEAT(ACCUM_FIFO_BODY)
        
                // -------------------------------
                // Assign values from output_buf into the partial sum registers for the first row of PEs
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                LABEL(INIT_OUT) for(int j = 0; j < OC0; ++j) {
                    psum_reg[0][j] = output_buf.value[j];
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------
            

                // -------------------------------
                // Run the 16x16 PE array
                // Make sure that the correct registers are given to the PE
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                LABEL(COL) for (int j=0; j < OC0; ++j) {
                    #pragma hls_unroll yes
                    LABEL(ROW) for (int i=0; i < IC0; ++i) {
                        pe[i][j].run(input_reg[i][j], psum_reg[i][j], weight_reg[i][j], input_reg2[i][j], psum_reg2[i][j]);
                    } //ROW
                } //COL
                // -------------------------------
                // Your code ends here
                // -------------------------------

                // Captures PE register state into log files
                #if HLS_DEBUG
                #ifndef __SYNTHESIS__
                log_matrix(input_file, input_reg, step, OC0);
                log_matrix(weight_file, weight_reg, step, OC0);
                log_matrix(psum_file, psum_reg, step, OC0);
                #endif
                #endif
                

                /*
                 * FIFOs for partial outputs coming out of the systolic array
                 * The skewed version will be in the variable output_row
                 */
                PackedInt<OUTPUT_PRECISION, OC0> output_row;

                #define FIFO_WRITE_BODY_NEW(z,i,unused)\
                    ODTYPE BOOST_PP_CAT(accum_fifo_output_, i); \
                    BOOST_PP_CAT(accum_fifo_, i).run( psum_reg[IC0][i] , BOOST_PP_CAT(accum_fifo_output_, i) );\
                    output_row.value[i] = BOOST_PP_CAT(accum_fifo_output_,i); \
                
                REPEAT(FIFO_WRITE_BODY_NEW)

                // -------------------------------
                // After a certain number of cycles, you will have valid output from the systolic array
                // Depending on the loop indices, this valid output will either be written into the accumulation buffer or written out
                // Your code starts here
                // -------------------------------
                if(step >= OC0+IC0-1){
                    #pragma hls_unroll yes
                    for(int i = 0; i < OC0; i++){
                        accumulation_buffer[step-(IC0+OC0-1)][i] = output_row.value[i];
                    }
                    if (loopIndices.ic1_idx==params.IC1-1 && loopIndices.fx_idx == params.FX-1 && loopIndices.fy_idx == params.FY-1) {   
                        output.write(output_row);
                    }
                }
                // -------------------------------
                // Your code ends here
                // -------------------------------
                
                // -------------------------------
                // Cycle the input/psum registers
                // That is, the outputs that a PE wrote to should now become the input for the next PE
                // Your code starts here
                // -------------------------------
                #pragma hls_unroll yes
                for(int j = 0; j < OC0; j++){
                    #pragma hls_unroll yes
                    for(int i = 0; i < IC0; i++){
                        input_reg[i][j+1] = input_reg2[i][j];
                        psum_reg[i+1][j] = psum_reg2[i][j];
                    }
                }

                // -------------------------------
                // Your code ends here
                // -------------------------------
                if (step == step_bound-1) break;
            }
        }
    
        // Debug example:
        // printf("outputs written: %d\n", output.size());
    }

private:
    
    // -------------------------------
    // Create the following:
    //  - PE array
    //  - accumulation buffer
    //  - weight registers
    //  - input registers (two sets, one at the input of the PE and one at the output) 
    //  - psum registers (two sets, one at the input of the PE and one at the output) 
    // Your code starts here
    // -------------------------------
    ProcessingElement<IDTYPE, ODTYPE> pe[IC0][OC0];

    ODTYPE accumulation_buffer[ACCUMULATION_BUFFER_SIZE][OC0];
    WDTYPE weight_reg[IC0][OC0];
    IDTYPE input_reg[IC0][OC0+1];
    IDTYPE input_reg2[IC0][OC0];
    ODTYPE psum_reg[IC0+1][OC0];
    ODTYPE psum_reg2[IC0][OC0];
    // -------------------------------
    // Your code ends here
    // -------------------------------
    

#define INPUT_FIFOS_INIT(z, i, unused) \
    Fifo<IDTYPE, i + 1> BOOST_PP_CAT(input_fifo_, i);

    REPEAT(INPUT_FIFOS_INIT)

#define ACCUM_FIFOS_INIT(z, i, unused) \
    Fifo<ODTYPE, i + 1> BOOST_PP_CAT(psum_fifo_, i);

    REPEAT(ACCUM_FIFOS_INIT)
    

#define OUTPUT_FIFOS_INIT(z, i, unused) \
    Fifo<ODTYPE, OC0 - i> BOOST_PP_CAT(accum_fifo_, i);
    
    REPEAT(OUTPUT_FIFOS_INIT)
};

#endif
