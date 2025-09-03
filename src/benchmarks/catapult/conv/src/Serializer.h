#ifndef SERIALIZER_H
#define SERIALIZER_H

template<typename DTYPE, typename DTYPE_SERIAL, int OC0, int accumbuffersize>
class Serializer{
public:
    Serializer(){}

    #pragma hls_design interface
    #pragma hls_pipeline_init_interval 1
    void CCS_BLOCK(run)(ac_channel<DTYPE> &inputChannel,
                        ac_channel<DTYPE_SERIAL> &serialOutChannel,
                        ac_channel<Params> &paramsIn)
        {
            #ifndef __SYNTHESIS__
            while(inputChannel.available(1))
            #endif
            {
                // #ifndef __SYNTHESIS__
                // printf("ofmap out channel size: %d\n", inputChannel.size());
                // printf("paramsIn stream channel size: %d\n", paramsIn.size());
                // #endif
                Params params = paramsIn.read();
                uint_16 tile_size = params.OX0 * params.OY0;
                DTYPE_SERIAL buffer[accumbuffersize][OC0];

                for(int i = 0; i < OX0_MAX * OY0_MAX; i++){
                    DTYPE input = inputChannel.read();
                    #pragma hls_unroll yes
                    for(int j = 0; j < OC0; j++){
                        buffer[i][j] = input.value[j];
                    }

                    if (i == tile_size - 1) break;
                }

                for(int i = 0; i < OX0_MAX * OY0_MAX; i++){
                    for(int j = 0; j < OC0; j++){
                        serialOutChannel.write(buffer[i][j]);
                    }

                    if (i == tile_size - 1) break;
                }
            }
        }
    };


#endif
