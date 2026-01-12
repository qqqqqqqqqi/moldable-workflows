import time

import networkx as nx

from Base import util
from prop_mapping_strategy import PropMappingStrategy


class PMCkptCrossover(PropMappingStrategy):

    def __init__(self, w, G, P, p_fail, lam, gamma, eta, ccr):
        super().__init__(w, G, P, p_fail, lam, gamma, eta, ccr)

    def checkpoint(self):
        for v in self.chainOnP:
            position = 0  #
            while position != len(v):
                sub_chain = v[position]
                C = util.C(0, len(sub_chain), sub_chain, self.G)
                self.ckpt[sub_chain[-1]][0]=1
                self.ckpt[sub_chain[-1]][1]=C*self.gamma
                self.ckpt[sub_chain[-1]][2]=C*self.eta
                position+=1

    def get_schedule(self,outdir):
        start=time.perf_counter()
        self.decompose()
        self.Map(self.G, [0, self.P - 1])
        self.checkpoint()
        end = time.perf_counter()
        self.output2csv( f"{outdir}/1_{self.w}_{self.P}_{self.p_fail}_{self.ccr:.1e}.csv")
        return end-start,self.get_makespan()

    def get_makespan(self):
        graph = nx.DiGraph()
        original_edges = {}
        node_mapping = {}
        t_weight={}
        for v in self.chainOnP:
            position = 0  # start of position of G in PC
            merged_node = []
            merged_node_succ = []
            last_node = ''
            while position != len(v):
                sub_chain = v[position]
                weight=0
                for node in sub_chain:
                    weight+=self.G.nodes[node]['d_weight']
                    merged_node_succ += self.G.successors(node)
                    merged_node.append(node)
                new_node = sub_chain[-1]
                for node in merged_node:
                    node_mapping[node] = new_node
                original_edges[new_node] = list(set(merged_node_succ))
                graph.add_node(new_node)
                t_weight[new_node]=weight+sum(self.ckpt[new_node][1:3])
                if last_node != '' and last_node != new_node:
                    graph.add_edge(last_node, new_node)
                last_node = new_node
                merged_node = []
                merged_node_succ = []
                position += 1

        for key in original_edges.keys():
            for succ in original_edges[key]:
                if key != node_mapping[succ]:
                    graph.add_edge(key, node_mapping[succ])
        return util.compute_longest_weighted_path(graph,t_weight)


