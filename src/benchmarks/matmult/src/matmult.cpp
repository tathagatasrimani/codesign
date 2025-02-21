#include "matmult.h"
#include <mc_scverify.h>

#pragma hls_design top
class MatMult {
    public:
        MatMult(){}

        #pragma hls_design interface
        void CCS_BLOCK(run)(ac_channel<PackedInt2D<PRECISION, 5, 5> > &a_chan, 
                            ac_channel<PackedInt2D<PRECISION, 5, 5> > &b_chan,
                            ac_channel<PackedInt2D<PRECISION, 5, 5> > &c_chan)
        {
            #ifndef __SYNTHESIS__
            while (a_chan.available(1)) {
            #endif
                PackedInt2D<PRECISION, 5, 5> a = a_chan.read();
                PackedInt2D<PRECISION, 5, 5> b = b_chan.read();
                PackedInt2D<PRECISION, 5, 5> c;
                do_matmult(a, b, c);
                c_chan.write(c);
            #ifndef __SYNTHESIS__
            }
            #endif
        }
    private:
        #pragma hls_design ccore
        void do_matmult(PackedInt2D<PRECISION, 5, 5> &a,
                        PackedInt2D<PRECISION, 5, 5> &b,
                        PackedInt2D<PRECISION, 5, 5> &c) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    c.value[i].value[j] = 0;
                }
            }
            
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    ac_int<PRECISION> tmp = 0; 
                    for (int k = 0; k < 5; k++) {
                        tmp += a.value[i].value[k] * b.value[k].value[j];
                    }
                    c.value[i].value[j] = tmp;
                }
            }
        }
};
