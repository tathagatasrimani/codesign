#ifndef DESERIALIZER_H
#define DESERIALIZER_H

template<typename DTYPE_SERIAL, typename DTYPE, int n>
class Deserializer{
public:
    Deserializer(){}

#pragma hls_design interface
void CCS_BLOCK(run)(ac_channel<DTYPE_SERIAL> &inputChannel,
                    ac_channel<DTYPE> &outputChannel)
    {
        #ifndef __SYNTHESIS__
        while(inputChannel.available(1))
        #endif
        {
            DTYPE output;
            for(int i = 0; i < n; i++){
                output.value[i] = inputChannel.read();
            }
            outputChannel.write(output);

        }
    }
};

class ParamsDeserializer{
public:
    ParamsDeserializer(){}

#pragma hls_design interface
#pragma hls_pipeline_init_interval 9
void CCS_BLOCK(run)(ac_channel<uint_16> &inputChannel,
                    ac_channel<Params> &outputChannel1,
                    ac_channel<Params> &outputChannel2,
                    ac_channel<Params> &outputChannel3,
                    ac_channel<Params> &outputChannel4
                    )
    {
        Params params;
        
        params.OY1 = inputChannel.read();
        params.OX1 = inputChannel.read();
        params.OY0 = inputChannel.read();
        params.OX0 = inputChannel.read();
        params.OC1 = inputChannel.read();
        params.IC1 = inputChannel.read();
        params.FX = inputChannel.read();
        params.FY = inputChannel.read();
        params.STRIDE = inputChannel.read();

        outputChannel1.write(params);  // input double buffer
        outputChannel2.write(params);  // weight double buffer
        outputChannel3.write(params);  // systolic array
        for (int i = 0; i < OX1_MAX * OY1_MAX * OC1_MAX; i++) {
            outputChannel4.write(params);  // output serializer

            if (i == params.OX1 * params.OY1 * params.OC1 - 1) break;
        }
    }

};

#endif
