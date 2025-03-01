#include "conv.h"
#include "conv_gold_tiled.cpp"
#include "conv_gold.cpp"
#include "Conv.cpp"
#include "conv_tb_params.h"

template <int OFMAP_HEIGHT, 
          int OFMAP_WIDTH, 
          int OFMAP_CHANNELS, 
          int IFMAP_CHANNELS, 
          int FILTER_SIZE, 
          int STRIDE,
          int IC0,
          int OC0>
int run_layer(Params params){
    IDTYPE input[(OFMAP_HEIGHT-1)*STRIDE+FILTER_SIZE][(OFMAP_WIDTH-1)*STRIDE+FILTER_SIZE][IFMAP_CHANNELS]; 
    WDTYPE weight[FILTER_SIZE][FILTER_SIZE][IFMAP_CHANNELS][OFMAP_CHANNELS]; 
    ODTYPE output_ref[OFMAP_HEIGHT][OFMAP_WIDTH][OFMAP_CHANNELS];
    ODTYPE output_ref_tiled[OFMAP_HEIGHT][OFMAP_WIDTH][OFMAP_CHANNELS];

    static ac_channel<PackedInt<INPUT_PRECISION, 4> > input_stream;
    static ac_channel<PackedInt<WEIGHT_PRECISION, 4> > weight_stream;
    static ac_channel<ODTYPE> output_stream;
    
    int errCnt = 0;
    int rand_init = 1;

    printf("Generating Input\n");
 
    // initialize input image  
    for (int row = 0; row < STRIDE * (OFMAP_HEIGHT-1) + FILTER_SIZE; row++) {
      for (int col = 0; col < STRIDE * (OFMAP_WIDTH-1) + FILTER_SIZE; col++) {
        for (int c = 0; c < IFMAP_CHANNELS; c++) {
          if (rand_init == 1) {
            input[row][col][c] = (IDTYPE)(rand() % 100); 
          } else {
            input[row][col][c] = c + IFMAP_CHANNELS*col + IFMAP_CHANNELS*(OFMAP_WIDTH+FILTER_SIZE-1)*row;
          }
        }
      }
    }

    // streaming input to the interface
    for (int ro = 0; ro < params.OY1; ro++) {
      for (int co = 0; co < params.OX1; co++) {
        for (int c=0; c< params.IC1; c++) {
          for (int p = 0; p < STRIDE*(params.OY0-1) + FILTER_SIZE; p++ ){
            for (int j = 0; j < (STRIDE*(params.OX0-1) + FILTER_SIZE); j++ ){
              for (int i = 0; i < IC0/4; i++ ){
                PackedInt<INPUT_PRECISION, 4> input_tmp;
                for(int ii = 0; ii < 4; ii++){
                  input_tmp.value[ii] = input[ro*STRIDE*params.OY0+p][co*STRIDE*params.OX0+j][c*IC0+i*4+ii];
                }
                input_stream.write(input_tmp);
              }  // for i
            }  // for j 
          }  // for p
        }  // for c
      }  // for co
    }  // for ro
 

    printf("Generating Weight\n");

    // initialize weights
    for (int wy = 0; wy < FILTER_SIZE; wy++) {  
      for (int wx = 0; wx < FILTER_SIZE; wx++) {  
        for (int c = 0; c < IFMAP_CHANNELS; c++) {
          for (int k = 0; k < OFMAP_CHANNELS; k++) {
            if (rand_init == 1) {
              weight[wy][wx][c][k] = (IDTYPE)(rand()%100);  
            } else {
              weight[wy][wx][c][k] = c + k + OFMAP_CHANNELS*c + OFMAP_CHANNELS*IFMAP_CHANNELS*wx + OFMAP_CHANNELS*IFMAP_CHANNELS*FILTER_SIZE*wy;  
            }
          }
        }  
      }
    }
    
    printf("Streaming Weight\n");
    // streaming weight to the interface
    for (int ro = 0; ro < params.OY1; ro++) {
      for (int co = 0; co < params.OX1; co++) {     
        for(int koo = 0; koo < params.OC1; koo++){
          for (int c = 0; c < params.IC1; c++) {
            for (int wy = 0; wy <params.FY; wy++) {
              for (int wx = 0; wx <params.FX; wx++) {
                for ( int i = 0; i < IC0; i++ ){
                    for ( int j = 0; j < OC0/4; j++ ){
                      PackedInt<WEIGHT_PRECISION, 4> weight_tmp;
                      for(int jj = 0; jj < 4; jj++){
                        weight_tmp.value[jj] = weight[wy][wx][c*IC0+i][koo*OC0 + j*4+jj];
                      }
                      weight_stream.write(weight_tmp);
                    }  // for j
                }  // for i
              }  // for wy
            }  // for wx
          }  // for k
        } // for koo
      }  // for co
    }  // for ko 


    static ac_channel<uint_16> params_stream;
    params_stream.write(params.OY1);
    params_stream.write(params.OX1);
    params_stream.write(params.OY0);
    params_stream.write(params.OX0);
    params_stream.write(params.OC1);
    params_stream.write(params.IC1);
    params_stream.write(params.FX);
    params_stream.write(params.FY);
    params_stream.write(params.STRIDE);

    // Main function call
    // launch hardware design
    // conv *conv_design = new conv;
    printf("Running HLS C design\n");
    Conv conv_design;
    conv_design.run(input_stream,weight_stream,output_stream, params_stream); 

    printf("Running reference C models\n");
    // run reference model
    conv_gold_tiled<IDTYPE,ODTYPE,OFMAP_HEIGHT,OFMAP_WIDTH,OFMAP_CHANNELS,IFMAP_CHANNELS,FILTER_SIZE,STRIDE>(params.OY1,  params.OY0,  params.OX1,  params.OX0,  params.OC1,  OC0,  params.IC1,  IC0,  params.FX,  params.FY, input, weight, output_ref_tiled);          
    conv_gold<IDTYPE,ODTYPE,OFMAP_HEIGHT,OFMAP_WIDTH,OFMAP_CHANNELS,IFMAP_CHANNELS,FILTER_SIZE,STRIDE>(input, weight, output_ref);          

    printf("\nChecking Output\n\n"); 
    // compare the hardware results with the reference model
    for (int ro = 0; ro < params.OY1; ro++) {
      for (int co = 0; co < params.OX1; co++) {
        for(int koo = 0; koo < params.OC1; koo++){
          for (int p = 0; p < params.OY0; p++ ){
            for (int i = 0; i < params.OX0; i++ ){

              for (int j = 0; j < OC0; j++) {
                
               ODTYPE out_value = output_stream.read();

                if ((long long)output_ref[ro*params.OY0+p][co*params.OX0+i][koo*OC0+j] != (long long)output_ref_tiled[ro*params.OY0+p][co*params.OX0+i][koo*OC0+j]) {
                  printf("***REFERENCE ERROR***\n");
                  printf("output[%d][%d][%d], ref = %lld, ref tiled = %lld\n",ro*params.OY0+p, co*params.OX0+i, koo*OC0+j, (long long)output_ref[ro*params.OY0+p][co*params.OX0+i][koo*OC0+j], (long long)output_ref_tiled[ro*params.OY0+p][co*params.OX0+i][koo*OC0+j]);
                }

                if((long long)output_ref[ro*params.OY0+p][co*params.OX0+i][koo*OC0+j] != (long long)out_value) {
                  errCnt++;
                  if (errCnt < 10) {
                    printf("***ERROR***\n");
                    printf("output[%d][%d][%d] = %lld, ref = %lld\n",ro*params.OY0+p, co*params.OX0+i, koo*OC0+j, (long long)out_value, (long long)output_ref[ro*params.OY0+p][co*params.OX0+i][koo*OC0+j]);
                  }
                }
              }  // for j
            }  // for i
          }  // for p
        } // for koo
      }  // for co
    }  // for ko
    
    printf("\nThere were %d errors\n", errCnt);
    return errCnt;
}

CCS_MAIN(int argc, char *argv[]) 
{
    int errCnt = 0;
    
    Params params_resnet_layer = {
        OY1,
        OX1,
        OY0,
        OX0,
        OC1,
        IC1,
        FX,
        FY,
        STRIDE
    };
    errCnt += run_layer<OY0 * OY1, OX0 * OX1, OC0 * OC1, IC0 * IC1, FX, STRIDE, IC0, OC0>(params_resnet_layer);
    
   //   printf("Layer 1\n");
   //   Params params_resnet_layer_1 = {
   //       8, // OY1
   //       8, // OX1
   //       14, // OY0
   //       14, // OX0
   //       4, // OC1 
   //       1, // IC1
   //       7, // FX
   //       7, // FY
   //       2 // STRIDE
   //   };
   //   errCnt += run_layer<112, 112, 64, 64, 7, 2, 16, 16>(params_resnet_layer_1);

    // printf("Layer 2\n");
    // Params params_resnet_layer_2 = {
    //     4, // OY1
    //     4, // OX1
    //     14, // OY0
    //     14, // OX0
    //     4, // OC1 
    //     4, // IC1
    //     3, // FX
    //     3, // FY
    //     1 // STRIDE
    // };
    // errCnt += run_layer<56, 56, 64, 64, 3, 1, 16, 16>(params_resnet_layer_2);
   
    //  printf("Layer 3_1\n");
    //  Params params_resnet_layer_3_1 = {
    //      2, // OY1
    //      2, // OX1
    //      14, // OY0
    //      14, // OX0
    //      8, // OC1 
    //      4, // IC1
    //      3, // FX
    //      3, // FY
    //      2 // STRIDE
    //  };
    //  errCnt += run_layer<28, 28, 128, 64, 3, 2, 16, 16>(params_resnet_layer_3_1);

    //  printf("Layer 3\n");
    //  Params params_resnet_layer_3 = {
    //      2, // OY1
    //      2, // OX1
    //      14, // OY0
    //      14, // OX0
    //      8, // OC1 
    //      8, // IC1
    //      3, // FX
    //      3, // FY
    //      1 // STRIDE
    //  };
    //  errCnt += run_layer<28, 28, 128, 128, 3, 1, 16, 16>(params_resnet_layer_3);
      
    //  printf("Layer 4_1\n");
    //  Params params_resnet_layer_4_1 = {
    //      2, // OY1
    //      2, // OX1
    //      7, // OY0
    //      7, // OX0
    //      16, // OC1 
    //      8, // IC1
    //      3, // FX
    //      3, // FY
    //      2 // STRIDE
    //  };
    //  errCnt += run_layer<14, 14, 256, 128, 3, 2, 16, 16>(params_resnet_layer_4_1);

    //  printf("Layer 4\n");
    //  Params params_resnet_layer_4 = {
    //      1, // OY1
    //      1, // OX1
    //      14, // OY0
    //      14, // OX0
    //      16, // OC1 
    //      16, // IC1
    //      3, // FX
    //      3, // FY
    //      1 // STRIDE
    //  };
    //  errCnt += run_layer<14, 14, 256, 256, 3, 1, 16, 16>(params_resnet_layer_4);

    //  printf("Layer 5_1\n");
    //  Params params_resnet_layer_5_1 = {
    //      1, // OY1
    //      1, // OX1
    //      7, // OY0
    //      7, // OX0
    //      32, // OC1 
    //      16, // IC1
    //      3, // FX
    //      3, // FY
    //      2 // STRIDE
    //  };
    //  errCnt += run_layer<7, 7, 512, 256, 3, 2, 16, 16>(params_resnet_layer_5_1);

    //  printf("Layer 5\n");
    // Params params_resnet_layer_5 = {
    //     1, // OY1
    //     1, // OX1
    //     7, // OY0
    //     7, // OX0
    //     32, // OC1 
    //     32, // IC1
    //     3, // FX
    //     3, // FY
    //     1 // STRIDE
    // };
    // errCnt += run_layer<7, 7, 512, 512, 3, 1, 16, 16>(params_resnet_layer_5);
    
    if (errCnt == 0) {
      CCS_RETURN(0);
    } else {
      CCS_RETURN(1);
    }
}
