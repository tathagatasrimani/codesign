#!/bin/sh
BENCHMARK_DIR=benchmarks/models

./simulate.sh $BENCHMARK_DIR/spmv.py spmv.py $1
./simulate.sh $BENCHMARK_DIR/aes.py aes.py $1
./simulate.sh $BENCHMARK_DIR/hpcg.py hpcg.py $1
./simulate.sh $BENCHMARK_DIR/genomics.py genomics.py $1
./simulate.sh $BENCHMARK_DIR/dijkstra.py dijkstra.py $1

if [ $2 ]; then
    ./simulate.sh $BENCHMARK_DIR/bert.py bert.py $1
    ./simulate.sh $BENCHMARK_DIR/resnet18.py resnet18.py $1
fi