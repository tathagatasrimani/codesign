#include <cstdio>
#include <mc_scverify.h>
#include "conv.h"
#include "InputDoubleBuffer.h"
#include <vector>
#include <fstream>
#include "conv_tb_params.h"

bool pcompare(PackedInt<INPUT_PRECISION, IC0> expected, PackedInt<INPUT_PRECISION, IC0> actual) {
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
    IDTYPE input[(OFMAP_HEIGHT-1)*STRIDE+FILTER_SIZE][(OFMAP_WIDTH-1)*STRIDE+FILTER_SIZE][IFMAP_CHANNELS]; 

    static ac_channel<PackedInt<INPUT_PRECISION, 4> > inputs_in_stream;
    static ac_channel<PackedInt<INPUT_PRECISION, IC0> > inputs_out_stream;
    static ac_channel<Params> params_stream;
    
    int errCnt = 0;

    printf("Generating Input\n");
 
    // initialize input image  
    for (int row = 0; row < STRIDE * (OFMAP_HEIGHT-1) + FILTER_SIZE; row++) {
      for (int col = 0; col < STRIDE * (OFMAP_WIDTH-1) + FILTER_SIZE; col++) {
        for (int c = 0; c < IFMAP_CHANNELS; c++) {
            input[row][col][c] = c + IFMAP_CHANNELS*col + IFMAP_CHANNELS*(OFMAP_WIDTH+FILTER_SIZE-1)*row;
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
                inputs_in_stream.write(input_tmp);
              }  // for i
            }  // for j 
          }  // for p
        }  // for c
      }  // for co
    }  // for ro

    params_stream.write(params);

    // Run HLS
    printf("Running HLS C design\n");
    InputDoubleBuffer<INPUT_BUFFER_SIZE, IC0, OC0> inputdoublebuffer_dut;
    inputdoublebuffer_dut.run(inputs_in_stream, inputs_out_stream, params_stream); 

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
    std::vector<PackedInt<INPUT_PRECISION, IC0>> gold_inputs;
    std::vector<int> numbers;

    // Read in numbers
    int number;
    while (goldfile >> number) {
        numbers.push_back(number);
    }

    // Pack numbers
    for (int i = 0; i < numbers.size(); i += IC0) {
        PackedInt<INPUT_PRECISION, IC0> tmp;
        for (int j = 0; j < IC0; j++) {
            tmp.value[j] = numbers[i + j];
        }
        gold_inputs.push_back(tmp);
    }

    printf("Gold inputs size %d\n", gold_inputs.size());

    printf("\nChecking Output\n\n"); 
    // Compare the gold results with the actual model
    for (PackedInt<INPUT_PRECISION, IC0> input_expected: gold_inputs) {
        PackedInt<INPUT_PRECISION, IC0> input_actual = inputs_out_stream.read();
        if (!pcompare(input_expected, input_actual)) {
              errCnt++;
              if (errCnt < 10) {
                printf("***ERROR***\n");
                printf("Expected = %s\nActual = %s\n", input_actual.to_string().c_str(), input_expected.to_string().c_str());
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
