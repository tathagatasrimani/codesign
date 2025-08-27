
to parse the verbose.rpt dump:
python3 parse_verbose_rpt.py resnet_18/resnet18_sol/.autopilot/db/ test_parse

to create the CDFG: 
python3 create_cdfg.py test_parse

to create the merged CDFG: 
python3 merge_cdfgs.py test_parse forward


Current status: 

python3 merge_cdfgs.py test_parse_3 dataflow_in_loop_VITIS_LOOP_5877_1 successfully creates CDFG