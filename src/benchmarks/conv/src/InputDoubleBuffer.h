#ifndef INPUT_DOUBLE_BUFFER_H
#define INPUT_DOUBLE_BUFFER_H


template <int size, int IC0, int OC0>
class InputDoubleBufferWriter{
public:
    InputDoubleBufferWriter(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<PackedInt<INPUT_PRECISION, 4> > &din,
                        ac_channel<chanStruct<PackedInt<INPUT_PRECISION,IC0>,size> > &dout)
    {
        #ifndef __SYNTHESIS__
        while (din.available(1)) {
        #endif
            // -------------------------------
            // Your code starts here
            // -------------------------------
            // #ifndef __SYNTHESIS__
            // printf("input stream channel size: %d\n", din.size());
            // printf("paramsIn stream channel size: %d\n", paramsIn.size());
            // #endif
        
            Params params = paramsIn.read();
            ac_int<ac::log2_ceil<size+1>::val, false> tileSize = ((params.OX0 - 1) * params.STRIDE + params.FX) * 
                                ((params.OY0 - 1) * params.STRIDE + params.FY) * 
                                params.IC1;
            
            chanStruct<PackedInt<INPUT_PRECISION,IC0>,size> tmp;

            // record one tile in buffer
            #pragma hls_pipeline_init_interval 2
            TILE: for (int i = 0; i < IX0_MAX * IY0_MAX * IC1_MAX; i++) {
                PackedInt<INPUT_PRECISION, IC0> memCol;  // one column in the memory
                // each packet contains 4 values, pack IC0 tgt into one row
                for (int j = 0; j < IC0; j=j+4) {
                    PackedInt<INPUT_PRECISION, 4> packet = din.read();

                    #pragma hls_unroll yes
                    for (int k = 0; k < 4; k++) {
                        memCol.value[j+k] = packet.value[k];
                    }
                }
                tmp.data[i] = memCol;

                if (i==tileSize-1) break;
            } // TILE
            // write one tile
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
class InputDoubleBufferReader{
public:
    InputDoubleBufferReader(){}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<Params> &paramsIn,
                        ac_channel<chanStruct<PackedInt<INPUT_PRECISION, IC0>,size> > &din, 
                        ac_channel<PackedInt<INPUT_PRECISION, IC0> > &dout)
    {
        #ifndef __SYNTHESIS__
        while (din.available(1)) {
        #endif
            // -------------------------------
            // Your code starts here
            // -------------------------------
            // #ifndef __SYNTHESIS__
            // printf("input tile channel size: %d\n", din.size());
            // printf("paramsIn stream channel size: %d\n", paramsIn.size());
            // #endif

            Params params = paramsIn.read();
            uint_16 IX0 = (params.OX0 - 1) * params.STRIDE + params.FX;
            uint_16 IY0 = (params.OY0 - 1) * params.STRIDE + params.FY;
            chanStruct<PackedInt<INPUT_PRECISION, IC0>,size> tmp;
            
            // read one tile from memory, and pass out one address at a time in the correct order
            tmp = din.read();
            // OC1 reuses
            #pragma hls_pipeline_init_interval 1
            OC1: for (int oc1 = 0; oc1 < OC1_MAX; oc1++) {
                IC1: for (int ic1 = 0; ic1 < IC1_MAX; ic1++) {
                    FY: for (int fy = 0; fy < FY_MAX; fy++) {
                        FX: for (int fx = 0; fx < FX_MAX; fx++) {
                            OY0: for (int oy0 = 0; oy0 < OY0_MAX; oy0++) { 
                                OX0: for (int ox0 = 0; ox0 < OX0_MAX; ox0++) { 
                                    uint_16 address = 
                                            params.STRIDE * ox0 + fx +
                                            (params.STRIDE * oy0 + fy) * IX0 +
                                            IY0 * IX0 * ic1;
                                    dout.write(tmp.data[address]);

                                    if (ox0 == params.OX0-1) break;
                                } // OX0

                                if(oy0 == params.OY0-1) break;
                            } // OY0

                            if(fx == params.FX-1) break;
                        } // FX

                        if(fy == params.FY-1) break;
                    } // FY

                    if(ic1 == params.IC1-1) break;
                } // IC1

                if(oc1 == params.OC1-1) break;
            } // OC1
            // -------------------------------
            // Your code ends here
            // -------------------------------
        #ifndef __SYNTHESIS__
        }
        #endif
    }
};

template <int size, int IC0, int OC0>
class InputDoubleBuffer{
public:
  InputDoubleBuffer(){}

  #pragma hls_design interface
  void CCS_BLOCK(run)(ac_channel<PackedInt<INPUT_PRECISION, 4> > &inputs_in, 
                      ac_channel<PackedInt<INPUT_PRECISION, IC0> > &inputs_out,
                      ac_channel<Params> &paramsIn)
    {

        Params params = paramsIn.read();

        for (int i = 0; i < OX1_MAX * OY1_MAX; i++) {
            inputDoubleBufferReaderParams.write(params);
            inputDoubleBufferWriterParams.write(params);
            if (i == params.OX1 * params.OY1 - 1) break;
        }

        inputDoubleBufferWriter.run(inputDoubleBufferWriterParams, inputs_in, mem);

        inputDoubleBufferReader.run(inputDoubleBufferReaderParams, mem, inputs_out);
    }

private:
    ac_channel<chanStruct<PackedInt<INPUT_PRECISION, IC0>,size> > mem;
    
    InputDoubleBufferWriter<size, IC0, OC0> inputDoubleBufferWriter;
    ac_channel<Params> inputDoubleBufferWriterParams;
    
    InputDoubleBufferReader<size, IC0, OC0> inputDoubleBufferReader;
    ac_channel<Params> inputDoubleBufferReaderParams;
};

#endif
