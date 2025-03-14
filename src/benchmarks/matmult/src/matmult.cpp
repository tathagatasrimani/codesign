//#include "c_cores.h"  // Include the CCORE header
#include "ccores/add.h"  // Include the adder CCORE
#include "ccores/mult.h"  // Include the multiplier CCORE
#include "matmult.h"
#include <mc_scverify.h>

#pragma hls_design top
class MatMult {
    add add_inst[TILE_SIZE][TILE_SIZE];  // Instantiate the adder blackbox
    mult mul_inst[TILE_SIZE][TILE_SIZE];  // Instantiate the multiplier blackbox
    public:
        MatMult(){}
        #pragma hls_design interface
        void CCS_BLOCK(run)(ac_channel<PackedInt2D<PRECISION, 10, 10> > &a_chan, 
                            ac_channel<PackedInt2D<PRECISION, 10, 10> > &b_chan,
                            ac_channel<PackedInt2D<PRECISION, 10, 10> > &c_chan)
        {
            #ifndef __SYNTHESIS__
            while (a_chan.available(1)) {
            #endif
                PackedInt2D<PRECISION, 10, 10> a = a_chan.read();
                PackedInt2D<PRECISION, 10, 10> b = b_chan.read();
                PackedInt2D<PRECISION, 10, 10> c;
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> add_2;
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> add_out;
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> mul_1;
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> mul_2;
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> mul_out;
                for (int i = 0; i < TILE_SIZE; i++) {
                    for (int j = 0; j < TILE_SIZE; j++) {
                        mul_inst[i][j].run(mul_1.value[i].value[j], mul_2.value[i].value[j], mul_out.value[i].value[j]);
                        add_inst[i][j].run(mul_out.value[i].value[j], add_2.value[i].value[j], add_out.value[i].value[j]);
                    }
                }
                //#pragma hls_unroll yes
                for (int i = 0; i < 10; i++) {
                    //#pragma hls_unroll yes
                    for (int j = 0; j < 10; j++) {
                        c.value[i].value[j] = 0;
                    }
                }
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> c_tmp;
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> a_tmp;
                PackedInt2D<PRECISION, TILE_SIZE, TILE_SIZE> b_tmp;
                for (int i = 0; i < 10; i+= TILE_SIZE) {
                    for (int j = 0; j < 10; j+= TILE_SIZE) {
                        for (int k = 0; k < 10; k+= TILE_SIZE) {
                            for (int ii = 0; ii < TILE_SIZE; ii++) {
                                for (int jj = 0; jj < TILE_SIZE; jj++) {
                                    a_tmp.value[ii].value[jj] = a.value[ii+i].value[jj+k];
                                    b_tmp.value[ii].value[jj] = b.value[ii+k].value[jj+j];
                                    c_tmp.value[ii].value[jj] = c.value[ii+i].value[jj+j];
                                }
                            }
                            
                            for (int kk = 0; kk < TILE_SIZE; kk++) {
                                #pragma hls_unroll yes
                                for (int ii = 0; ii < TILE_SIZE; ii++) {
                                    #pragma hls_unroll yes
                                    for (int jj = 0; jj < TILE_SIZE; jj++) {
                                        mul_1.value[ii].value[jj] = a_tmp.value[ii].value[kk];
                                        mul_2.value[ii].value[jj] = b_tmp.value[kk].value[jj];
                                        add_2.value[ii].value[jj] = c_tmp.value[ii].value[jj];
                                        c_tmp.value[ii].value[jj] = add_out.value[ii].value[jj];
                                    }
                                }
                            }

                            for (int ii = 0; ii < TILE_SIZE; ii++) {
                                for (int jj = 0; jj < TILE_SIZE; jj++) {
                                    c.value[ii+i].value[jj+j] = c_tmp.value[ii].value[jj];
                                }
                            }
                        }
                    }
                }
                c_chan.write(c);
            #ifndef __SYNTHESIS__
            }
            #endif
        }
    private:
};