import time

import networkx as nx
from networkx.utils import flatten

from Base import util
from prop_mapping_strategy import PropMappingStrategy


class PMCkptAll(PropMappingStrategy):

    def __init__(self, w, G, P, p_fail, lam, gamma, eta, ccr):
        super().__init__(w, G, P, p_fail, lam, gamma, eta, ccr)


    def checkpoint(self):
        for node in self.G.nodes:
            fs = {node: self.G.nodes[node]['ck_ext']}
            for succ in self.G.successors(node):
                fs.update(self.G.edges[node, succ]['filelist'])
            self.ckpt[node][0] = 1
            self.ckpt[node][1] = sum(fs.values()) * self.gamma
            self.ckpt[node][2] = sum(fs.values()) * self.eta

    def get_schedule(self,outdir):
        start=time.perf_counter()
        self.decompose()
        self.Map(self.G, [0, self.P - 1])
        self.checkpoint()
        end=time.perf_counter()
        self.output2csv( f"{outdir}/0_{self.w}_{self.P}_{self.p_fail}_{self.ccr:.1e}.csv")
        return end-start,self.get_makespan()

    def get_makespan(self):
        graph = nx.DiGraph()
        nodeOn1P=[]
        t_weight={}
        for v in self.chainOnP:
            nodeOn1P.append(flatten(v))
        for v in nodeOn1P:
            position = 0  # start of position of G in P
            last_node = ''
            while position != len(v):
                current_node = v[position]
                weight = self.G.nodes[current_node]['d_weight']
                C = self.ckpt[current_node][1]
                V =  self.ckpt[current_node][2]
                graph.add_node(current_node)
                t_weight[current_node] = weight+C+V
                if last_node != '':
                    graph.add_edge(last_node, current_node)
                last_node = current_node
                position += 1
        graph.add_edges_from(self.G.edges())
        return util.compute_longest_weighted_path(graph,t_weight)