from collections import deque
# format: cfg_node -> {states -> operations}
node_operations = {}
operation_sets = {}

def schedule(cfg, graphs, benchmark):
    for node in cfg:
        #print(node.id)
        node_operations[node] = []
        operation_sets[node] = set()
        graph = graphs[node]
        queue = deque([[root, 0] for root in graph.roots])
        max_order = 0
        while len(queue) != 0:
            cur_node, order = queue.popleft()
            if cur_node not in operation_sets[node]: operation_sets[node].add(cur_node)
            max_order = max(max_order, order)
            cur_node.order = max(order, cur_node.order)
            for child in cur_node.children:
                queue.append([child, order+1])
        for i in range(max_order+1):
            node_operations[node].append([])
        for cur_node in operation_sets[node]:
            node_operations[node][cur_node.order].append(cur_node)
        """for state in node_operations[node]:
            for op in state:
                print(op.order, op.operation)
            print('')"""
    return cfg, node_operations
        

if __name__ == '__main__':
    schedule("")