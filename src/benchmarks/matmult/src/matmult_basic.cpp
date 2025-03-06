#include "matmult.h"
#include <mc_scverify.h>

#pragma hls_design top
class MatMult {
    public:
        MatMult(){}

        #pragma hls_design interface
        void CCS_BLOCK(run)(ac_channel<PackedInt2D<PRECISION, 100, 100> > &a_chan, 
                            ac_channel<PackedInt2D<PRECISION, 100, 100> > &b_chan,
                            ac_channel<PackedInt2D<PRECISION, 100, 100> > &c_chan)
        {
            #ifndef __SYNTHESIS__
            while (a_chan.available(1)) {
            #endif
                PackedInt2D<PRECISION, 100, 100> a = a_chan.read();
                PackedInt2D<PRECISION, 100, 100> b = b_chan.read();
                PackedInt2D<PRECISION, 100, 100> c;
                #pragma hls_pipeline_init_interval 1
                //#pragma hls_unroll yes
                for (int i = 0; i < 100; i++) {
                    //#pragma hls_unroll yes
                    for (int j = 0; j < 100; j++) {
                        c.value[i].value[j] = 0;
                    }
                }
                #pragma hls_pipeline_init_interval 1
                //#pragma hls_unroll yes
                for (int i = 0; i < 100; i++) {
                    //#pragma hls_unroll yes
                    for (int j = 0; j < 100; j++) {
                        ac_int<PRECISION> tmp = 0;
                        //#pragma hls_pipeline_init_interval 1
                        for (int k = 0; k < 100; k++) {
                            tmp += a.value[i].value[k] * b.value[k].value[j];
                        }
                        c.value[i].value[j] = tmp;
                    }
                }
                c_chan.write(c);
            #ifndef __SYNTHESIS__
            }
            #endif
        }
    private:
};
