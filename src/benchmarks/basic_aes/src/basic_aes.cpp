#include "ccores/add.h"
#include "ccores/mult.h"
#include "ccores/bitxor.h"
#include <ac_channel.h>
#include <ac_int.h>
#include <mc_scverify.h>

#define BLOCK_SIZE 16  // AES block size in bytes
#define PRECISION 8    // 8-bit data for AES
#define NUM_BLOCKS 4   // <-- compile-time constant number of blocks
#define ROUNDS 10      // <-- compile-time constant number of rounds

#pragma hls_design top
class basic_aes {
    add add_inst;
    mult mul_inst;
    bitxor xor_inst;

public:
    basic_aes() {}

    #pragma hls_design interface
    void CCS_BLOCK(run)(ac_channel<ac_int<PRECISION>> &data_in_chan,
                        ac_channel<ac_int<PRECISION>> &key_chan,
                        ac_channel<ac_int<PRECISION>> &data_out_chan)
    {
        #ifndef __SYNTHESIS__
        while (data_in_chan.available(BLOCK_SIZE * NUM_BLOCKS)) {
        #endif
            // Load key
            ac_int<PRECISION> key[BLOCK_SIZE];
            for (int i = 0; i < BLOCK_SIZE; i++) {
                key[i] = key_chan.read();
            }

            // Process each block
            for (int b = 0; b < NUM_BLOCKS; b++) {
                ac_int<PRECISION> block[BLOCK_SIZE];

                // Load data block
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    block[i] = data_in_chan.read();
                }

                // Do AES-like rounds
                for (int r = 0; r < ROUNDS; r++) {
                    // SubBytes + AddRoundKey (here just xor with key using CCORE)
                    #pragma hls_unroll yes
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        ac_int<16,true> a_ext = block[i];
                        ac_int<16,true> k_ext = key[i % BLOCK_SIZE];
                        //ac_int<16,true> t_ext = b * BLOCK_SIZE * ROUNDS + r * BLOCK_SIZE + i;
                        ac_int<16,true> tmp16;
                        xor_inst.run(a_ext, k_ext, tmp16);
                        block[i] = tmp16.slc<8>(0);
                    }

                    // ShiftRows
                    ac_int<PRECISION> tmp;
                    tmp = block[1]; block[1] = block[5]; block[5] = block[9]; block[9] = block[13]; block[13] = tmp;
                    tmp = block[2]; block[2] = block[10]; block[10] = tmp;
                    tmp = block[6]; block[6] = block[14]; block[14] = tmp;
                    tmp = block[3]; block[3] = block[15]; block[15] = block[11]; block[11] = block[7]; block[7] = tmp;

                    // MixColumns: simple xor + adds replaced by CCOREs
                    #pragma hls_unroll yes
                    for (int c = 0; c < 4; c++) {
                        int base = c * 4;
                        ac_int<PRECISION> a = block[base];
                        ac_int<PRECISION> b_ = block[base+1];
                        ac_int<PRECISION> c_ = block[base+2];
                        ac_int<PRECISION> d = block[base+3];

                        ac_int<16,true> a_ext = a;
                        ac_int<16,true> b_ext = b_;
                        ac_int<16,true> c_ext = c_;
                        ac_int<16,true> d_ext = d;
                        //ac_int<16,true> tag_base = base * 10 + r;

                        ac_int<16,true> ab, bc, cd, da;
                        // xor_inst.run(a_ext, b_ext, tag_base, ab);
                        // xor_inst.run(b_ext, c_ext, tag_base+1, bc);
                        // xor_inst.run(c_ext, d_ext, tag_base+2, cd);
                        // xor_inst.run(d_ext, a_ext, tag_base+3, da);

                        xor_inst.run(a_ext, b_ext, ab);
                        xor_inst.run(b_ext, c_ext, bc);
                        xor_inst.run(c_ext, d_ext, cd);
                        xor_inst.run(d_ext, a_ext, da);

                        ac_int<16,true> abm, bcm, cdm, dam;
                        // mul_inst.run(ab, 1, tag_base+10, abm);
                        // mul_inst.run(bc, 1, tag_base+11, bcm);
                        // mul_inst.run(cd, 1, tag_base+12, cdm);
                        // mul_inst.run(da, 1, tag_base+13, dam);
                        mul_inst.run(ab, 1, abm);
                        mul_inst.run(bc, 1, bcm);
                        mul_inst.run(cd, 1, cdm);
                        mul_inst.run(da, 1, dam);

                        block[base]   = abm.slc<8>(0);
                        block[base+1] = bcm.slc<8>(0);
                        block[base+2] = cdm.slc<8>(0);
                        block[base+3] = dam.slc<8>(0);
                    }

                    // Final AddRoundKey again with XOR CCORE
                    #pragma hls_unroll yes
                    for (int i = 0; i < BLOCK_SIZE; i++) {
                        ac_int<16,true> a_ext = block[i];
                        ac_int<16,true> k_ext = key[i % BLOCK_SIZE];
                        //ac_int<16,true> t_ext = b * BLOCK_SIZE * ROUNDS + r * BLOCK_SIZE + i + 2000;
                        ac_int<16,true> tmp16;
                        xor_inst.run(a_ext, k_ext, tmp16);
                        block[i] = tmp16.slc<8>(0);
                    }
                }

                // Write block to output channel
                for (int i = 0; i < BLOCK_SIZE; i++) {
                    data_out_chan.write(block[i]);
                }
            }
        #ifndef __SYNTHESIS__
        }
        #endif
    }
};
