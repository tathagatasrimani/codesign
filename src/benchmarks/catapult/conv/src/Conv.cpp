#ifndef CONV_TOP_CPP
#define CONV_TOP_CPP


#ifdef __SYNTHESIS__
    #define LABEL(x) x:
#else
    #define LABEL(x) {}
#endif

#include "conv.h"
#include <mc_scverify.h>

#include "Serializer.h"
#include "Deserializer.h"

#include "InputDoubleBuffer.h"
#include "WeightDoubleBuffer.h"
#include "SystolicArray.h"


#pragma hls_design top
class Conv{
public:
    Conv(){}

#pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<PackedInt<INPUT_PRECISION, 4> > &input_serial, 
                        ac_channel<PackedInt<WEIGHT_PRECISION, 4> > &weight_serial, 
                        ac_channel<ODTYPE> &output_serial,
                        ac_channel<uint_16> &paramsIn)
    {
        paramsDeserializer.run(paramsIn, inputDoubleBufferParams, weightDoubleBufferParams, systolicArrayParams, outputSerializerParams);

        inputDoubleBuffer.run(input_serial, input_out, inputDoubleBufferParams);
        weightDoubleBuffer.run(weight_serial, weight_out, weightDoubleBufferParams);
        systolicArray.run(input_out, weight_out, output, systolicArrayParams);

        outputSerializer.run(output, output_serial, outputSerializerParams);   
    }

private:
    ParamsDeserializer paramsDeserializer;
    Serializer<PackedInt<OUTPUT_PRECISION, ARRAY_DIMENSION>, ODTYPE, ARRAY_DIMENSION, ACCUMULATION_BUFFER_SIZE> outputSerializer;
    ac_channel<Params> outputSerializerParams;

    InputDoubleBuffer<INPUT_BUFFER_SIZE, ARRAY_DIMENSION, ARRAY_DIMENSION> inputDoubleBuffer;
    ac_channel<Params> inputDoubleBufferParams;

    WeightDoubleBuffer<WEIGHT_BUFFER_SIZE, ARRAY_DIMENSION, ARRAY_DIMENSION> weightDoubleBuffer;
    ac_channel<Params> weightDoubleBufferParams;
    
    ac_channel<PackedInt<INPUT_PRECISION,ARRAY_DIMENSION> > input_out;
    ac_channel<PackedInt<WEIGHT_PRECISION,ARRAY_DIMENSION> > weight_out;
    ac_channel<PackedInt<OUTPUT_PRECISION,ARRAY_DIMENSION> > output;    

    SystolicArrayWrapper<IDTYPE,WDTYPE,ODTYPE, ARRAY_DIMENSION, ARRAY_DIMENSION> systolicArray;
    ac_channel<Params> systolicArrayParams;
};

#endif
