#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include "ProcessingElement.h"
#include "conv.h"
#include "Fifo.h"
#include "SystolicArrayCore.h"

// Include mc_scverify.h for CCS_* macros
#include <mc_scverify.h>

class SystolicArrayLooper
{
public:
    SystolicArrayLooper() {}

#pragma hls_design interface
#pragma hls_pipeline_init_interval 1
void run(ac_channel<Params> &paramsIn,
         ac_channel<Params> &paramsOut,
         ac_channel<LoopIndices> &loopIndicesOut)
    {
        // -------------------------------
        // Generate the loop indices here for the systolic array.
        // Write the loop indices as well as the params out to channels.
        // Your code starts here
        // -------------------------------
        Params params = paramsIn.read();

        LABEL(OX1) for (int oy1 = 0; oy1 < OY1_MAX; oy1++) {
            LABEL(OY1) for (int ox1 = 0; ox1 < OX1_MAX; ox1++) {  
                LABEL(OC1) for(uint_16 oc1 = 0; oc1 < OC1_MAX; ++oc1){    
                    LABEL(IC1) for (uint_16 ic1 = 0; ic1 < IC1_MAX; ++ic1) {
                        LABEL(FY) for (uint_16 fy = 0; fy < FY_MAX; ++fy) { 
                            LABEL(FX) for (uint_16 fx = 0; fx < FX_MAX; ++fx) { 
                                LoopIndices loopIndices = {
                                    ic1, 
                                    fx, 
                                    fy
                                };
                                loopIndicesOut.write(loopIndices);
                                paramsOut.write(params);

                                if (fx == params.FX-1) break;
                            } // FX

                            if (fy == params.FY-1) break;
                        } // FY

                        if (ic1 == params.IC1-1) break;
                    } // IC1

                    if (oc1 == params.OC1-1) break;
                } // OC1

                if (ox1 == params.OX1-1) break;
            } // OX1

            if (oy1 == params.OY1-1) break;
        } // OY1
        // -------------------------------
        // Your code ends here
        // -------------------------------
    }
};

template <typename IDTYPE, typename WDTYPE, typename ODTYPE, int OC0, int IC0>
class SystolicArrayWrapper
{
public:
    SystolicArrayWrapper(){}
    
#pragma hls_design interface
#pragma hls_pipeline_init_interval 1
    void run(ac_channel<PackedInt<INPUT_PRECISION, IC0> > &input, 
             ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &weight, 
             ac_channel<PackedInt<OUTPUT_PRECISION, OC0> > &output,
             ac_channel<Params> &paramsIn)
    {
        systolicArrayLooper.run(paramsIn, paramsChannel, loopIndicesChannel);
        systolicArrayCore.run(input, weight, output, paramsChannel, loopIndicesChannel);
    }
private:
    SystolicArrayCore<IDTYPE, WDTYPE, ODTYPE, OC0, IC0> systolicArrayCore;
    SystolicArrayLooper systolicArrayLooper;
    ac_channel<Params> paramsChannel;
    ac_channel<LoopIndices> loopIndicesChannel;
};

#endif
