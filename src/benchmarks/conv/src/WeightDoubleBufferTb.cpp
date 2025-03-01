#include <cstdio>
#include <mc_scverify.h>
#include "conv.h"
#include "WeightDoubleBuffer.h"
#include <vector>
#include <fstream>
#include "conv_tb_params.h"

bool pcompare(PackedInt<WEIGHT_PRECISION, OC0> expected, PackedInt<WEIGHT_PRECISION, OC0> actual) {
    bool match = true;
    for (int i = 0; i < OC0; i++) {
        if (expected.value[i] != actual.value[i]) {
            match = false;
        }
    }
    return match;
}

template <int OFMAP_HEIGHT, 
          int OFMAP_WIDTH, 
          int OFMAP_CHANNELS, 
          int IFMAP_CHANNELS, 
          int FILTER_SIZE, 
          int STRIDE,
          int IC0,
          int OC0>
int run_layer(Params params) {
    WDTYPE weight[FILTER_SIZE][FILTER_SIZE][IFMAP_CHANNELS][OFMAP_CHANNELS]; 

    static ac_channel<PackedInt<WEIGHT_PRECISION, 4> > weights_in_stream;
    static ac_channel<PackedInt<WEIGHT_PRECISION, OC0> > weights_out_stream;
    static ac_channel<Params> params_stream;
    
    int errCnt = 0;

    printf("Generating Weight\n");

    // initialize weights
    for (int wy = 0; wy < FILTER_SIZE; wy++) {  
      for (int wx = 0; wx < FILTER_SIZE; wx++) {  
        for (int c = 0; c < IFMAP_CHANNELS; c++) {
          for (int k = 0; k < OFMAP_CHANNELS; k++) {
            weight[wy][wx][c][k] = c + k + OFMAP_CHANNELS*c + OFMAP_CHANNELS*IFMAP_CHANNELS*wx + OFMAP_CHANNELS*IFMAP_CHANNELS*FILTER_SIZE*wy;  
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
                      weights_in_stream.write(weight_tmp);
                    }  // for j
                }  // for i
              }  // for wy
            }  // for wx
          }  // for k
        } // for koo
      }  // for co
    }  // for ko 

    params_stream.write(params);

    // Run HLS
    printf("Running HLS C design\n");
    WeightDoubleBuffer<WEIGHT_BUFFER_SIZE, IC0, OC0> weightdoublebuffer_dut;
    weightdoublebuffer_dut.run(weights_in_stream, weights_out_stream, params_stream); 

    printf("Loading correct comparison\n");

    // Open the file
    char *compare_filename = getenv("COMPARE_FILE");
    std::ifstream goldfile(compare_filename); // TODO: parameterize this file

    // Check if the file is open
    if (!goldfile.is_open()) {
        std::cerr << "Error opening comparison file: " << std::string(compare_filename) << std::endl;
        return -1;
    }

    // Load correct test vector
    std::vector<PackedInt<WEIGHT_PRECISION, OC0>> gold_weights;
    std::vector<int> numbers;

    // Read in numbers
    int number;
    while (goldfile >> number) {
        numbers.push_back(number);
    }

    // Pack numbers
    for (int i = 0; i < numbers.size(); i += OC0) {
        PackedInt<WEIGHT_PRECISION, OC0> tmp;
        for (int j = 0; j < OC0; j++) {
            tmp.value[j] = numbers[i + j];
        }
        gold_weights.push_back(tmp);
    }

    printf("Gold weights size %d\n", gold_weights.size());

    printf("\nChecking Output\n\n"); 
    // Compare the gold results with the actual model
    for (PackedInt<WEIGHT_PRECISION, OC0> weight_expected: gold_weights) {
        PackedInt<WEIGHT_PRECISION, OC0> weight_actual = weights_out_stream.read();
        if (!pcompare(weight_expected, weight_actual)) {
              errCnt++;
              if (errCnt < 10) {
                printf("***ERROR***\n");
                printf("Expected = %s\nActual = %s\n", weight_actual.to_string().c_str(), weight_expected.to_string().c_str());
              }
        }
    }
    
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
    
    if (errCnt == 0) {
      CCS_RETURN(0);
    } else {
      CCS_RETURN(1);
    }
}
