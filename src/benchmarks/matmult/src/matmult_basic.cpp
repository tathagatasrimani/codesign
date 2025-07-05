//#include "c_cores.h"  // Include the CCORE header
#include "ccores/add.h"  // Include the adder CCORE
#include "ccores/mult.h"  // Include the multiplier CCORE
#include "matmult.h"
#include <mc_scverify.h>

#define MATRIX_SIZE 4

#pragma hls_design top
class MatMult { 
    add add_inst;  // Instantiate the adder blackbox
    mult mul_inst;  // Instantiate the multiplier blackbox
    public:
        MatMult(){}

        #pragma hls_design interface
        void CCS_BLOCK(run)(ac_channel<PackedInt2D<PRECISION, MATRIX_SIZE, MATRIX_SIZE> > &a_chan, 
                            ac_channel<PackedInt2D<PRECISION, MATRIX_SIZE, MATRIX_SIZE> > &b_chan,
                            ac_channel<PackedInt2D<PRECISION, MATRIX_SIZE, MATRIX_SIZE> > &c_chan)
        {
            #ifndef __SYNTHESIS__
            while (a_chan.available(1)) {
            #endif
                PackedInt2D<PRECISION, MATRIX_SIZE, MATRIX_SIZE> a = a_chan.read();
                PackedInt2D<PRECISION, MATRIX_SIZE, MATRIX_SIZE> b = b_chan.read();
                PackedInt2D<PRECISION, MATRIX_SIZE, MATRIX_SIZE> c;

                //#pragma hls_pipeline_init_interval 1
                #pragma hls_unroll yes
                for (int i = 0; i < MATRIX_SIZE; i++) {
                    #pragma hls_unroll yes
                    for (int j = 0; j < MATRIX_SIZE; j++) {
                        c.value[i].value[j] = 0;
                    }
                }
                //#pragma hls_pipeline_init_interval 1
                #pragma hls_unroll no
                for (int i = 0; i < MATRIX_SIZE; i++) {
                    #pragma hls_unroll no
                    for (int j = 0; j < MATRIX_SIZE; j++) {
                        ac_int<PRECISION> tmp = 0;
                        //#pragma hls_pipeline_init_interval 1
                        #pragma hls_unroll no
                        for (int k = 0; k < MATRIX_SIZE; k++) {
                            // tmp += a.value[i].value[k] * b.value[k].value[j];
                            // Use CCOREs for multiplication and addition
                            // ac_int<PRECISION> product;
                            
                            // product = multiplier(a.value[i].value[k], b.value[k].value[j]);
                            
                            // tmp = adder(tmp, product);

                            ac_int<PRECISION> product;
                            ac_int<PRECISION> new_tmp;
                            ac_int<PRECISION> tag;

                            // tag needs to have i j and k to be unique
                            tag = i * MATRIX_SIZE * MATRIX_SIZE + j * MATRIX_SIZE + k;

                            // Perform multiplication using blackbox
                            mul_inst.run(a.value[i].value[k], b.value[k].value[j], tag, product);

                            // Perform addition using blackbox
                            add_inst.run(tmp, product, new_tmp);

                            tmp = new_tmp;  // Update tmp with the new sum
                        }
                        c.value[i].value[j] = tmp;
                    }
                }
                c_chan.write(c);
            #ifndef __SYNTHESIS__
            }
            #endif
        }
};