The purpose of this file is to rapidly evaluate different transistor model configurations at different sets of parameter values, and to see how the solver optimizes them over generations. 

To run, invoke the file from the codesign/ folder.

example command: 
python3 -m test.experiments.dennard_multi_core --config vitis_test_gemm --dummy True --tech_node <name> 

Args:
--config: sets up the entire codesign flow. Using different configs won't really matter for this flow so just set it to vitis_test_gemm

--tech_node: specifies which set of tech values to start from. Configurations are listed in src/yaml/tech_nodes.yaml and more can be added.

--dummy: won't invoke forward pass if True. Just always set this to true.

--obj: sets objective function for the solver to target. Examples are delay, edp, energy, ed2 (look at src/hardware_model/hardwareModel.py) (default edp, but I have used delay a lot)

-N: number of iterations for optimization flow (default 3)

--inverse_pass_improvement: default 10, sets the factor by which the solver tries to improve the system level objective in each generation.