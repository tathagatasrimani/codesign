#ifndef PROCESSING_ELEMENT_H
#define PROCESSING_ELEMENT_H

// Include mc_scverify.h for CCS_* macros
#include <mc_scverify.h>

template<typename IDTYPE, typename ODTYPE>
class ProcessingElement{
public:
    ProcessingElement(){}

#pragma hls_design interface ccore
    void CCS_BLOCK(run)(IDTYPE &input_in,
                        ODTYPE &psum_in,
                        WDTYPE &weight,
                        IDTYPE &input_out,
                        ODTYPE &psum_out)
    {
        // -------------------------------
        // Perform the MAC operation and forward inputs
        // Your code starts here
        // -------------------------------
        psum_out = input_in * weight + psum_in;
        input_out = input_in;
        // -------------------------------
        // Your code ends here
        // -------------------------------
    }
};

#endif
