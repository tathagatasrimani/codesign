import logging
logger = logging.getLogger(__name__)

import networkx as nx

class AbstractSimulator:

    def update_schedule_with_parasitics(self, scheduled_dfg):
        """
        Update edge weights in the scheduled data flow graph (DFG) after adding wire parasitics. This is
        used to ensure that the longest path calculation reflects the impact of parasitics.

        Args:
            scheduled_dfg (nx.DiGraph): Scheduled graph with nodes and edges representing the computation
                and their dependencies.

        Returns:
            None
        """
        for gen in list(nx.topological_generations(scheduled_dfg)):
            for node in gen:
                for parent in scheduled_dfg.predecessors(node):
                    edge = (parent, node)
                    scheduled_dfg.edges[edge]["weight"] = (scheduled_dfg.nodes[parent]["cost"] 
                                                            + scheduled_dfg.edges[edge]["cost"]) # update edge weight with parasitic
    
    def add_parasitics_to_scheduled_dfg(self, scheduled_dfg, parasitic_graph):
        """
        Add wire parasitics from OpenROAD to computation dfg
        params:
            scheduled_dfg: nx.DiGraph representing the scheduled graph
            parasitic_graph: nx.DiGraph representing wire parasitics from OpenROAD
        """

        def check_in_parasitic_graph(node_data):
            return ("allocation" in node_data
                and node_data["allocation"] != ""
                and "Mem" not in node_data["allocation"]
                and "Buf" not in node_data["allocation"])
        def accumulate_delay_over_segments(parasitic_edge):
            net_delay = 0
            res_instance = 0
            for y in range(len(parasitic_edge["net_cap"])): # doing second order RC
                res_instance += parasitic_edge["net_res"][y]
                cap_instance = parasitic_edge["net_cap"][y]
                net_delay += res_instance * cap_instance * 1e-3
            return net_delay
        def update_net_delay(node_name_prev, node_name):
            parasitic_edge = parasitic_graph[node_name_prev][node_name]
            net_delay = 0
            if isinstance(parasitic_edge["net_cap"], list):
                net_delay = accumulate_delay_over_segments(parasitic_edge)
            else:
                net_delay = (
                    parasitic_edge["net_cap"]
                    * parasitic_edge["net_res"]
                    * 1e-3
                )  # pico -> nano
            return net_delay
        # wire latency
        for edge in scheduled_dfg.edges:
            prev_node, node = edge
            node_data = scheduled_dfg.nodes[node]
            node_data_prev = scheduled_dfg.nodes[prev_node]
            node_name = node_data["allocation"]
            node_name_prev = node_data_prev["allocation"]
            net_delay = 0
            if (check_in_parasitic_graph(node_data) and 
                    check_in_parasitic_graph(node_data_prev)):  
                if "Regs" in node_name_prev or "Regs" in node_name: # again, 16 bit
                    max_delay = 0
                    #finding the longest time and adding that
                    for x in range(16):
                        if "Regs" in node_name_prev:
                            node_name_prev = node_name_prev + "_" + str(x)
                        elif "Regs" in node_name:
                            node_name = node_name + "_" + str(x)
                        if parasitic_graph.has_edge(node_name_prev, node_name):
                            net_delay = update_net_delay(node_name_prev, node_name)
                            if max_delay < net_delay:
                                max_delay = net_delay
                    net_delay = max_delay
                else: 
                    if parasitic_graph.has_edge(node_name_prev, node_name):
                        net_delay = update_net_delay(node_name_prev, node_name)
            scheduled_dfg.edges[edge]["cost"] = net_delay
        self.update_schedule_with_parasitics(scheduled_dfg)