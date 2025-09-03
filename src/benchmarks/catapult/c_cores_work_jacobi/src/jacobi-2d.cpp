
// #define N 100
//    static void kernel_jacobi_2d(int tsteps,
//                    int n,
//                    double A [N][N],
//                    double B [N][N])
//    {
   
//         for (int t = 0; t < tsteps; t++)
//         {
//             for (int i = 1; i < n - 1; i++){
//                 for (int j = 1; j < n - 1; j++){
//                     B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
//                 }
//             }
//             for (int i = 1; i < n - 1; i++) {
//                 for (int j = 1; j < n - 1; j++){
//                     A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
//                 }
//             }
//         }
//   
//    }

#include "add.h"  // Include the adder CCORE
#include "mult.h"  // Include the multiplier CCORE
#include "jacobi-2d.h"
#include <mc_scverify.h>

#define N 100
#define TSTEPS 100

#pragma hls_design top
class Jacobi2D { 
    add add_inst[8];  // Instantiate the adder blackbox
    mult mult_inst[2];  // Instantiate the multiplier blackbox
    public:
        Jacobi2D(){}

        #pragma hls_design interface
        void CCS_BLOCK(run)(ac_channel<PackedInt2D<PRECISION, N, N> > &a_chan, 
                            ac_channel<PackedInt2D<PRECISION, N, N> > &b_chan,
                            ac_channel<PackedInt2D<PRECISION, N, N> > &a_out_chan,
                            ac_channel<PackedInt2D<PRECISION, N, N> > &b_out_chan)
        {
            #ifndef __SYNTHESIS__
            while (a_chan.available(1)) {
            #endif
                PackedInt2D<PRECISION, N, N> a = a_chan.read();
                PackedInt2D<PRECISION, N, N> b = b_chan.read();
                PackedInt2D<PRECISION, N, N> a_out;
                PackedInt2D<PRECISION, N, N> b_out;


                #pragma hls_pipeline_init_interval 1
                //#pragma hls_unroll yes
                for (int i = 0; i < N; i++) {
                    //#pragma hls_unroll yes
                    for (int j = 0; j < N; j++) {
                        a_out.value[i].value[j] = a.value[i].value[j];
                        b_out.value[i].value[j] = b.value[i].value[j];
                    }
                }

                #pragma hls_pipeline_init_interval 8
                
                for (int t = 0; t < TSTEPS; t++)
                {
                    for (int i = 1; i < N - 1; i++) {
                        for (int j = 1; j < N - 1; j++) {
                            ac_int<PRECISION> sum, temp, temp1, temp2, temp3, temp4;

                            add_inst[0].run(a_out.value[i].value[j], a_out.value[i].value[j-1], temp1);
                            add_inst[1].run(temp1, a_out.value[i].value[j+1], temp2);
                            add_inst[2].run(temp2, a_out.value[i+1].value[j], temp3);
                            add_inst[3].run(temp3, a_out.value[i-1].value[j], sum);
                            mult_inst[0].run(sum, ac_int<PRECISION>(2), temp); // actually needs to be 0.2
                            
                            b_out.value[i].value[j] = temp;
                        }
                    }

                    for (int i = 1; i < N - 1; i++) {
                        for (int j = 1; j < N - 1; j++) {
                            ac_int<PRECISION> sum, temp, temp1, temp2, temp3, temp4;

                            add_inst[4].run(b_out.value[i].value[j], b_out.value[i].value[j-1], temp1);
                            add_inst[5].run(temp1, b_out.value[i].value[j+1], temp2);
                            add_inst[6].run(temp2, b_out.value[i+1].value[j], temp3);
                            add_inst[7].run(temp3, b_out.value[i-1].value[j], sum);
                            mult_inst[1].run(sum, ac_int<PRECISION>(2), temp); // actually needs to be 0.2
                            
                            a_out.value[i].value[j] = temp;
                        }
                    }
                }
 

                a_out_chan.write(a_out);
                b_out_chan.write(b_out);
            #ifndef __SYNTHESIS__
            }
            #endif
        }
};




 // for (int t = 0; t < TSTEPS; t++)
                // {
                //     for (int i = 1; i < N - 1; i++){
                //         for (int j = 1; j < N - 1; j++){
                //             b_out.value[i].value[j] = 0.2 * (a_out.value[i].value[j] + a_out.value[i].value[j-1] + a_out.value[i].value[1+j] + a_out.value[1+i].value[j] + a_out.value[i-1].value[j]);
                //         }
                //     }
                //     for (int i = 1; i < N - 1; i++) {
                //         for (int j = 1; j < N - 1; j++){
                //             a_out.value[i].value[j] = 0.2 * (b_out.value[i].value[j] + b_out.value[i].value[j-1] + b_out.value[i].value[1+j] + b_out.value[1+i].value[j] + b_out.value[i-1].value[j]);
                //         }
                //     }
                // }  