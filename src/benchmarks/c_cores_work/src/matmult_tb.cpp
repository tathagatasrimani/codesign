#include "matmult.cpp"
#include "matmult_gold.cpp"

int run_matmult() {
    DTYPE a[5][5];
    DTYPE b[5][5];
    DTYPE c[5][5];

    PackedInt2D<PRECISION, 5, 5> a_stream;
    PackedInt2D<PRECISION, 5, 5> b_stream;
    PackedInt2D<PRECISION, 5, 5> c_stream;

    static ac_channel<PackedInt2D<PRECISION, 5, 5> > a_chan;
    static ac_channel<PackedInt2D<PRECISION, 5, 5> > b_chan;
    static ac_channel<PackedInt2D<PRECISION, 5, 5> > c_chan;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            a[i][j] = (DTYPE)(rand()%100);
            a_stream.value[i].value[j] = a[i][j];
            b[i][j] = (DTYPE)(rand()%100);
            b_stream.value[i].value[j] = b[i][j];
        }
    }

    a_chan.write(a_stream);
    b_chan.write(b_stream);

    printf("Running HLS C Design\n");

    MatMult mm;
    mm.run(a_chan, b_chan, c_chan);

    printf("Running reference C model\n");

    matmult_gold<DTYPE, 5>(a, b, c);

    printf("Checking output\n");

    c_stream = c_chan.read();

    int errCnt = 0;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("reading %d from c_stream, %d from c\n", c_stream.value[i].value[j], c[i][j]);
            if (c_stream.value[i].value[j] != c[i][j]) errCnt++;
        }
    }

    return errCnt;
}


CCS_MAIN(int argc, char *argv[]) {
    int errCnt = 0;
    errCnt += run_matmult();
    if (errCnt == 0) {
        printf("No errors\n");
        CCS_RETURN(0);
    } else {
        printf("Errors detected!");
        CCS_RETURN(1);
    }
}