setenv MGLS_LICENSE_FILE 1717@cadlic0.stanford.edu
setenv PATH "${PATH}":/cad/mentor/2024.2/Mgc_home/bin/
setenv HOME "${PWD}"

module load incisive
#module load base
module load vitis/default


# for scalehls
# from scalehls-hida: setenv PATH "${PATH}":"${PWD}"/build/bin:"${PWD}"/polygeist/build/bin
# from samples/pytorch/resnet18: scalehls-opt resnet18.mlir -hida-pytorch-pipeline="top-func=forward loop-tile-size=8 loop-unroll-factor=4" | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives > resnet18.cpp
# from samples/polybench/gemm: cgeist test_gemm.c -function=test_gemm -S -memref-fullrank -raise-scf-to-affine > test_gemm.mlir
# from samples/polybench/gemm: scalehls-opt test_gemm.mlir -scalehls-dse-pipeline="top-func=test_gemm target-spec=../../../test/Transforms/Directive/config.json" | scalehls-translate -scalehls-emit-hlscpp > test_gemm_dse.cpp