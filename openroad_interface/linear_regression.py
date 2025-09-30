### TODO: This file needs to be updated to work with the refactored OpenRoad interface code. 

from . import openroad_run as pnr
# from . import validation as val


# dijkstra_detailed, _  = pnr.place_n_route(
#     "dijkstra.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
# )


# dijkstra_estimated, _  = pnr.place_n_route(
#     "dijkstra.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimated"
# )


# matmul_detailed, _ = pnr.place_n_route(
#     "matmul_80.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
# )
# matmul_estimate, _ = pnr.place_n_route(
#     "matmul_80.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
# )

# aladdin_const_with_mem_detailed, _ = pnr.place_n_route(
#     "aladdin_const_with_mem.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
# )
# aladdin_const_with_mem_estimate, _ = pnr.place_n_route(
#     "aladdin_const_with_mem.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
# )

# loop_detailed, _ = pnr.place_n_route(
#     "optimal_loop_unrolling.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
# )
# loop_estimated, _ = pnr.place_n_route(
#     "optimal_loop_unrolling.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
# )

aes_arch_detailed, _ = pnr.place_n_route(
    "aes_arch.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
)
aes_arch_estimated, _ = pnr.place_n_route(
    "aes_arch.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
)
# mm_test_detailed, _ = pnr.place_n_route(
#     "mm_test.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "detailed"
# )
# mm_test_estimation, _ = pnr.place_n_route(
#     "mm_test.gml", "openroad_interface/tcl/test_nangate45_bigger.tcl", "estimation"
# )

# dijkstra_data = val.pandas_organize("dijkstra.gml", dijkstra_estimated, dijkstra_detailed)
# aes_arch_data = val.pandas_organize("aes_arch.gml", aes_arch_estimated, aes_arch_detailed)
# mm_test_data = val.pandas_organize("mm_test.gml", mm_test_estimation, mm_test_detailed)