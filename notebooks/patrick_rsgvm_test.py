import networkx as nx
from src import hardwareModel
from src import architecture_search
from src import sim_util

benchmark = "src/benchmarks/models/add_and_mult.py"
config = "aladdin_const_with_mem"
sim, hw, computation_dfg = architecture_search.setup_arch_search(benchmark, config, gen_cacti=False)
hardwareModel.un_allocate_all_in_use_elements(hw.netlist)

parasitic_graph = nx.read_gml(sim_util.get_latest_log_dir()+"/parasitic_graph.gml")

scheduled_dfg = sim.schedule(computation_dfg, hw, "sdc")

longest_path_len_prev = nx.dag_longest_path_length(scheduled_dfg)

sim.add_parasitics_to_scheduled_dfg(scheduled_dfg, parasitic_graph)

longest_path_len = nx.dag_longest_path_length(scheduled_dfg)
longest_path = nx.dag_longest_path(scheduled_dfg)

print(longest_path_len_prev, longest_path_len)
print(longest_path)
