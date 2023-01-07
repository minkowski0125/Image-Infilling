import numpy as np

class Node:
    def __init__(self, id, weight):
        self.id = id
        self.weight = weight
        self.edges = []
    
    def add_egde(self, node_id):
        self.edges.append(node_id)

class Graph:
    def __init__(self, img, result_img, mask, x_0, x_1, y_0, y_1):
        self.img = img
        self.result_img = result_img
        self.mask = mask
        
        self.x_0 = x_0
        self.x_1 = x_1
        self.y_0 = y_0
        self.y_1 = y_1
        
        self.nodes = None
        self.edges = None
        self._build_graph()

    def _build_graph(self):
        x, y = self.mask.shape[:2]
        dif = np.linalg.norm(self.img - self.result_img, axis=-1)
        
        in_node = Node(-2, float('inf'))
        out_node = Node(-1, float('inf'))
        nodes, edges = {-2: in_node, -1: out_node}, {}
        total_weight = 0
        
        for i in range(self.x_0, self.x_1):
            for j in range(self.y_0, self.y_1):
                if self.mask[i][j] == 2:
                    new_node = Node(i * y + j, dif[i-self.x_0][j-self.y_0])
                    in_out_sign = 0
                    for (m, n) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                        if m < 0 or m >= x or n < 0 or n >= y:
                            continue
                        if self.mask[m][n] == 1:
                            in_out_sign = -2
                        elif self.mask[m][n] == 0:
                            in_out_sign = -1
                        else:
                            weight = dif[i-self.x_0][j-self.y_0] + dif[m-self.x_0][n-self.y_0]
                            new_node.add_egde(m * y + n)
                            edges[(i * y + j, m * y + n)] = weight
                            total_weight += weight
                    if in_out_sign < 0:
                        new_node.add_egde(in_out_sign)
                        nodes[in_out_sign].add_egde(i * y + j)
                        edges[(i * y + j, in_out_sign)] = edges[(in_out_sign, i * y + j)] = float('inf')
                        
                    nodes[i * y + j] = new_node
        self.nodes = nodes
        self.edges = edges
        self.total_weight = total_weight
                
    def get_edge_weight(self, p, q, tree):
        if tree == 2:
            return self.edges[(p, q)]
        elif tree == 1:
            return self.edges[(q, p)]
        else:
            raise NotImplementedError
                     
    def graph_cut_tree(self):
        print('doing graph cut...')
        s, t, a, o = [-2], [-1], [-2, -1], []
        parent, tree = {}, {-2:2, -1:1}
        flow_sum = 0
        
        while True:
            print('cutting on', flow_sum, self.total_weight, end='\r')
            
            # growth
            # print('grow')
            while len(a) > 0:
                node_id = a[0]
                node = self.nodes[node_id]
                
                find_path = 0
                for next_node_id in node.edges:
                    weight = self.get_edge_weight(node_id, next_node_id, tree[node_id])
                    if weight == 0:
                        continue
                    if not next_node_id in tree:
                        tree[next_node_id] = tree[node_id]
                        parent[next_node_id] = node_id
                        a.append(next_node_id)
                    elif tree[next_node_id] != tree[node_id]:
                        paths = []
                        for cur_id in [node_id, next_node_id]:
                            path = [cur_id]
                            while cur_id >= 0:
                                cur_id = parent[cur_id]
                                path.append(cur_id)
                            paths.append(path)
                        if cur_id == -2:
                            path = paths[1][::-1] + paths[0]
                        else:
                            path = paths[0][::-1] + paths[1]
                        find_path = 1
                        break
                if find_path:
                    break
                else:
                    a = a[1:]
                    path = None
            if path is None:
                break
                        
            # augment
            flow = min([self.edges[(path[i], path[i+1])] for i in range(len(path)-1)])
            flow_sum += flow
            for i in range(len(path) - 1):
                self.edges[(path[i], path[i+1])] -= flow
                self.edges[(path[i+1], path[i])] += flow
                if self.edges[(path[i], path[i+1])] == 0:
                    if tree[path[i]] == tree[path[i+1]] == 2:
                        parent.pop(path[i+1])
                        o.append(path[i+1])
                    elif tree[path[i]] == tree[path[i+1]] == 1:
                        parent.pop(path[i])
                        o.append(path[i])
            
            # adopt
            # print('adopt')
            while len(o) > 0:
                orphan_id = o[0]
                orphan_node = self.nodes[orphan_id]
                o = o[1:]
                adopted = 0
                for p_id in orphan_node.edges:
                    if tree.get(p_id, None) != tree[orphan_id]:
                        continue
                    weight = self.get_edge_weight(p_id, orphan_id, tree[orphan_id])
                    if weight == 0:
                        continue
                    cur_id, sign = p_id, 0
                    while cur_id >= 0:
                        if cur_id in o or not cur_id in tree or not cur_id in parent:
                            sign = 1
                            break
                        cur_id = parent[cur_id]
                    if sign:
                        continue
                    parent[orphan_id] = p_id
                    adopted = 1
                    break
                if not adopted:
                    for p_id in orphan_node.edges:
                        if tree.get(p_id, None) != tree[orphan_id]:
                            continue
                        weight = self.get_edge_weight(p_id, orphan_id, tree[orphan_id])
                        # if weight > 0:
                        #     if p_id in a:
                        #         print('unawared1!!!')
                        if weight > 0 and not p_id in a:
                            # print('add', p_id)
                            a.append(p_id)
                        if parent.get(p_id, None) == orphan_id:
                            if p_id in o:
                                print('unawared2!!!')
                            o.append(p_id)
                            parent.pop(p_id)
                    tree.pop(orphan_id)
                    # print('pop', orphan_id)
                    try:
                        orphan_idx = a.index(orphan_id)
                        a = a[:orphan_idx] + a[orphan_idx+1:]
                    except:
                        # print('unawared3!!!')
                        pass
        print('graph cut done.')
                  
    def graph_cut_bfs(self):
        print('doing graph cut...')
        node_queue, pre_weight, pre_node = [-2], {-2: float('inf')}, {}
        flow_sum = 0
        while True:
            print('cutting on', flow_sum, self.total_weight, end='\r')
            node_id = node_queue[0]
            node_queue = node_queue[1:]
            node = self.nodes[node_id]
            for i, next_node_id in enumerate(node.edges):
                if next_node_id in pre_node:
                    continue
                weight = self.edges[(node_id, next_node_id)]
                if weight == 0:
                    continue
                node_queue.append(next_node_id)
                pre_weight[next_node_id] = min(weight, pre_weight[node_id])
                pre_node[next_node_id] = node_id
                if next_node_id == -1:
                    cur_id = next_node_id
                    min_weight = pre_weight[cur_id]
                    flow_sum += min_weight
                    while cur_id != -2:
                        prev_id = pre_node[cur_id]
                        self.edges[(prev_id, cur_id)] -= min_weight
                        self.edges[(cur_id, prev_id)] += min_weight
                        cur_id = prev_id
                    node_queue, pre_weight, pre_node = [-2], {-2: float('inf')}, {}
                    break 
            if len(node_queue) == 0:
                break
                        
    def graph_cut(self):
        print('doing graph cut...')
        node_queue, ranks, closed_nodes = [-2], [], []
        prev_r = 0
        flow_sum = 0
        while True:
            node_id = node_queue[-1]
            # print('cutting on', node_id, len(node_queue), prev_r, flow_sum, self.total_weight, end='\r')
            print('cutting on', flow_sum, self.total_weight, end='\r')
            node = self.nodes[node_id]
            suc = 0
            for r in range(prev_r, len(node.edges)):
                next_node_id = node.edges[r]
                if next_node_id in node_queue or next_node_id in closed_nodes:
                    continue
                if self.edges[(node_id, next_node_id)] == 0:
                    continue
                suc = 1
                node_queue.append(next_node_id)
                ranks.append(r)
                prev_r = 0
                break
            if not suc:
                if node_queue[-1] == -2:
                    break
                closed_nodes.append(node_queue.pop())
                prev_r = ranks.pop() + 1
            elif node_queue[-1] == -1:
                # print('cutting', node_queue)
                flow = min([self.edges[(node_queue[i], node_queue[i+1])] for i in range(len(node_queue)-1)])
                flow_sum += flow
                for i in range(len(node_queue)-1):
                    self.edges[(node_queue[i], node_queue[i+1])] -= flow
                    self.edges[(node_queue[i+1], node_queue[i])] += flow
                node_queue, ranks, closed_nodes = [-2], [], []
                prev_r = 0     
    
    def generate_mask(self):
        i = 0
        x, y = self.mask.shape[:2]
        node_queue = [-2]
        mask = np.zeros_like(self.mask)
        while i < len(node_queue):
            node_id = node_queue[i]
            i += 1
            node = self.nodes[node_id]
            for next_node_id in node.edges:
                if next_node_id in node_queue:
                    continue
                if self.edges[(node_id, next_node_id)] == 0:
                    continue
                
                node_queue.append(next_node_id)
                mask[next_node_id // y, next_node_id % y] = 1
        return mask
            
                
def graph_cut(img, result_img, mask, dilated_mask, x_0, x_1, y_0, y_1):
    combined_mask = mask + dilated_mask * 2
    graph = Graph(img, result_img, combined_mask, x_0, x_1, y_0, y_1)
    graph.graph_cut_tree()
    mask = graph.generate_mask()
    return mask
    