import time

import networkx as nx
from networkx.utils import flatten

from Base import util
from prop_mapping_strategy import PropMappingStrategy


class PMCkptNone(PropMappingStrategy):

    def __init__(self, w, G, P, p_fail, lam, gamma, eta, ccr):
        super().__init__(w, G, P, p_fail, lam, gamma, eta, ccr)

    def checkpoint(self):
        for node in self.G.nodes:
            if self.G.out_degree(node) == 0:
                ck_ext = self.G.nodes[node]['ck_ext']
                self.ckpt[node] = [0, ck_ext * self.gamma, ck_ext * self.eta]

    def get_schedule(self, outdir):
        start=time.perf_counter()
        self.decompose()
        self.Map(self.G, [0, self.P - 1])
        self.checkpoint()
        end = time.perf_counter()
        self.output2csv(f"{outdir}/6_{self.w}_{self.P}_{self.p_fail}_{self.ccr:.1e}.csv")
        return end-start,self.get_makespan()

    def get_makespan(self):
        graph = nx.DiGraph()
        nodeOn1P = []
        t_weight = {}
        for v in self.chainOnP:
            nodeOn1P.append(flatten(v))
        for v in nodeOn1P:
            position = 0  # start of position of G in P
            last_node = ''
            while position != len(v):
                current_node = v[position]
                weight = self.G.nodes[current_node]['d_weight']
                graph.add_node(current_node)
                t_weight[current_node] = weight
                if last_node != '':
                    graph.add_edge(last_node, current_node)
                last_node = current_node
                position += 1
        graph.add_edges_from(self.G.edges())
        return util.compute_longest_weighted_path(graph, t_weight)