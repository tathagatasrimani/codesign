#include <iostream>
#include "attention.h"
using namespace std;

int main() {
    const int batch_size = 2;
    const int num_heads = 2;

    float Q[MAX_BATCH][MAX_HEADS][D_HEAD];
    float K_new[MAX_BATCH][MAX_HEADS][D_HEAD];
    float V_new[MAX_BATCH][MAX_HEADS][MAX_DVALUE];
    float output[MAX_BATCH][MAX_HEADS][MAX_DVALUE];

    attn_init(batch_size);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < D_HEAD; i++) {
                Q[b][h][i] = (float)(b + h + i);
                K_new[b][h][i] = (float)(b + h + i + 1);
                V_new[b][h][i] = (float)(b + h + i + 2);
            }
        }
    }

    attn_step(batch_size, num_heads, Q, K_new, V_new, output);

    cout << "Output from attn_step:\n";
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < D_HEAD; i++) {
                cout << output[b][h][i] << " ";
            }
            cout << endl;
        }
        cout << "---" << endl;
    }

    return 0;
}

