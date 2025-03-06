#include "basic_loops.h"
#include <mc_scverify.h>

#pragma hls_design top
class BasicLoops {
    public:
        BasicLoops(){}
        #pragma hls_design interface
        void CCS_BLOCK(run)(ac_channel<int>& a_c,
                            ac_channel<int>& c_c)
        {
            #ifndef __SYNTHESIS__
            while (a_c.available(1)) {
            #endif
                int a = a_c.read();
                int b = 0;
                for (int i = 0; i < 10; i++) {
                    a += 4;
                }
                int c = a;
                for (int i = 0; i < 8; i++) {
                    b += 5;
                    for (int j = 0; j < 8; j++) {
                        c += 6;
                    }
                }
                c_c.write(c+b);
            #ifndef __SYNTHESIS__
            }
            #endif
        }
    private:
};