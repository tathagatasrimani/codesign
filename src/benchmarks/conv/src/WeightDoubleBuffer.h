#ifndef WEIGHT_DOUBLE_BUFFER_H
#define WEIGHT_DOUBLE_BUFFER_H


template <int size, int IC0, int OC0>
class WeightDoubleBufferWriter{
public:
    WeightDoubleBufferWriter(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<PackedInt<WEIGHT_PRECISION, 4> > &din,
                        ac_channel<chanStruct<PackedInt<WEIGHT_PRECISION, OC0>, size> > &dout)
    {
        #ifndef __SYNTHESIS__
        while (din.available(1)) {
        #endif
            // -------------------------------
            // Your code starts here
            // -------------------------------
            // #ifndef __SYNTHESIS__
            // printf("weight stream channel size: %d\n", din.size());
            // printf("paramsIn stream channel size: %d\n", paramsIn.size());
            // #endif
        
            Params params = paramsIn.read();
            ac_int<ac::log2_ceil<size+1>::val, false> tileSize = params.FX * params.FY * IC0 * params.IC1;
            chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> tmp;

            #pragma hls_pipeline_init_interval 1
            TILE: for (int i = 0; i < FX_MAX * FY_MAX * IC0_MAX * IC1_MAX; i++) {
                // each packet contains 4 values, pack OC0 tgt into one row
                PackedInt<WEIGHT_PRECISION, OC0> memRow;  // one row in the memory
                for (int j = 0; j < OC0; j=j+4) {
                    PackedInt<WEIGHT_PRECISION, 4> packet = din.read();

                    #pragma hls_unroll yes
                    for (int k = 0; k < 4; k++) {
                        memRow.value[j+k] = packet.value[k];
                    }
                }
                tmp.data[i] = memRow;

                if (i == tileSize-1) break;
            }  // TILE
            dout.write(tmp);

            // -------------------------------
            // Your code ends here
            // -------------------------------
        #ifndef __SYNTHESIS__
        }
        #endif
    }
};

template <int size, int IC0, int OC0>
class WeightDoubleBufferReader{
public:
    WeightDoubleBufferReader(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> > &din, 
                        ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &dout)
    {
        #ifndef __SYNTHESIS__
        while (din.available(1)) {
        #endif
            // -------------------------------
            // Your code starts here
            // -------------------------------
            // #ifndef __SYNTHESIS__
            // printf("weight tile channel size: %d\n", din.size());
            // printf("paramsIn stream channel size: %d\n", paramsIn.size());
            // #endif
            
            Params params = paramsIn.read();
            ac_int<ac::log2_ceil<size+1>::val, false> tileSize = params.FX * params.FY * IC0 * params.IC1;

            chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> tmp;
            // read in new tile for every oc1
            tmp = din.read();
            #pragma hls_pipeline_init_interval 1
            TILE: for (int i = 0; i < FX_MAX * FY_MAX * IC0_MAX * IC1_MAX; i++) {
                dout.write(tmp.data[i]);

                if (i == tileSize-1) break;
            }
            // -------------------------------
            // Your code ends here
            // -------------------------------
        #ifndef __SYNTHESIS__
        }
        #endif
    }
};

template <int size, int IC0, int OC0>
class WeightDoubleBuffer{
public:
  WeightDoubleBuffer(){}

  #pragma hls_design interface
  void CCS_BLOCK(run)(ac_channel<PackedInt<WEIGHT_PRECISION, 4> > &weights_in, 
                      ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > &weights_out,
                      ac_channel<Params> &paramsIn)
    {
        Params params = paramsIn.read();

        for (int i = 0; i < OX1_MAX * OY1_MAX * OC1_MAX; i++) {
            weightDoubleBufferReaderParams.write(params);
            weightDoubleBufferWriterParams.write(params);
            if (i == params.OX1 * params.OY1 * params.OC1 - 1) break;
        }

        weightDoubleBufferWriter.run(weightDoubleBufferWriterParams, weights_in, mem);
        weightDoubleBufferReader.run(weightDoubleBufferReaderParams, mem, weights_out);
    }

private:
    ac_channel<chanStruct<PackedInt<WEIGHT_PRECISION, OC0>,size> > mem;
    
    WeightDoubleBufferWriter<size, IC0, OC0> weightDoubleBufferWriter;
    ac_channel<Params> weightDoubleBufferWriterParams;
    
    WeightDoubleBufferReader<size, IC0, OC0> weightDoubleBufferReader;
    ac_channel<Params> weightDoubleBufferReaderParams;
};


#endif
