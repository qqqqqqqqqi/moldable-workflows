import math
import time

import networkx as nx

from Base import util
from PropMappingAmdahl.prop_mapping_strategy import PropMappingStrategy


class PMCkptRight(PropMappingStrategy):

    def __init__(self, w, G, P, p_fail, lam, gamma, eta, ccr):
        super().__init__(w, G, P, p_fail, lam, gamma, eta, ccr)
        self.dp_time=0

    def checkpoint(self):
        for v in self.chainOnP:
            for sub_chain in v:
                assign_p=self.assign[sub_chain[0]]
                alloc=self.alloc[sub_chain[0]]
                ss=time.perf_counter()
                ckpt=self.ckpt_chain(sub_chain,assign_p)
                ee=time.perf_counter()
                self.dp_time+=(ee-ss)
                for k in ckpt.keys():
                    self.ckpt[k]=[1,ckpt[k][0],ckpt[k][1]]
                s=0
                e=0
                while e<len(sub_chain):
                    if ckpt.keys().__contains__(sub_chain[e]):
                        runtime,best_p=self.reduce(sub_chain[s:e+1],assign_p)
                        if best_p==self.assign[sub_chain[s]]:
                            s=e+1
                            e=s
                            continue
                        sub_p=assign_p- best_p
                        removed=alloc[-sub_p:]
                        r_alloc=alloc[:best_p]
                        for nd in sub_chain[s:e+1]:
                            self.assign[nd]=best_p
                            self.alloc[nd]=r_alloc
                            self.G.nodes[nd]['d_weight']=util.amdahl(self.G.nodes[nd]['weight'],self.G.nodes[nd]['alpha'],best_p)
                            for pp in removed:
                                self.schedule[pp].remove(nd)
                        s = e + 1
                        e = s
                    else:
                        e+=1


    def get_schedule(self,outdir):
        self.decompose()
        self.Map(self.G, [0, self.P - 1])
        self.checkpoint()
        self.output2csv( f"{outdir}/4_{self.w}_{self.P}_{self.p_fail}_{self.ccr:.1e}.csv")


    def get_makespan(self):
        graph = nx.DiGraph()
        original_edges = {}
        node_mapping = {}
        t_weight={}
        ckpt={nd:[self.ckpt[nd][1],self.ckpt[nd][2]] for nd in self.ckpt if self.ckpt[nd][0]==1}
        for v in self.chainOnP:
            position = 0  # start of position of G in PC
            merged_node = []
            merged_node_succ = []
            last_node = ''
            v=sum(v,[])
            while position != len(v):
                current_node = v[position]
                merged_node.append(current_node)
                merged_node_succ += self.G.successors(current_node)
                if ckpt.__contains__(current_node):
                    # if ckpt.has_key(PC[key][position]):
                    C = ckpt[current_node][0]  # the value of C and R when there is complicated dependency
                    V = ckpt[current_node][1]
                    new_node = current_node
                    weight=0
                    for node in merged_node:
                        node_mapping[node] = new_node
                        weight+=self.G.nodes[node]['d_weight']
                    original_edges[new_node] = list(set(merged_node_succ))
                    graph.add_node(new_node)
                    t_weight[new_node]=weight+C+V
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

    def reduce(self,L, max_p):
        bestP = 1
        f = util.C(0, len(L), L, self.G)
        V = f * self.eta
        C = f * self.gamma
        R = util.R(0, len(L), L, self.G) * self.gamma
        ws = [self.G.nodes[node]['weight'] for node in L]
        alphas = [self.G.nodes[node]['alpha'] for node in L]
        runtime = util.amdahlM(ws, alphas, 1)
        minE = math.exp(self.lam * runtime) * (runtime + V + R) + C - R
        for i in range(2, max_p + 1):
            nw = util.amdahlM(ws, alphas, i)
            E = math.exp(self.lam * i * nw) * (nw + V + R) + C - R
            if E < minE:
                minE = E
                bestP = i
                runtime = nw
        return runtime, bestP

    def ckpt_chain(self,L,assign_p):  # checkpointnew linear chain
        Exp = [[[math.inf for k in range(4)] for i in range(0, len(L) + 1)] for j in range(0, len(L) + 1)]
        Exp[0][0] = [0, 0, 0, 0]
        ckpt = {}
        prefix_w, prefix_r, prefix_c = util.pre_compute_topology(L, self.G)
        for i in range(1, len(L) + 1):
            for j in range(0, i):
                # w, rv = util.T(j, i, L, self.G)
                # c = util.C(j, i, L, self.G)
                w = prefix_w[i] - prefix_w[j]
                rv = prefix_r[j][i]
                c = prefix_c[j][i]
                v = c
                c *= self.gamma
                v *= self.eta
                rv *= self.gamma
                exp_weight = math.exp(self.lam * assign_p * w) * (w + v + rv) + c - rv
                Exp[i][j][1] = c
                Exp[i][j][2] = v
                Exp[i][j][3] = rv
                Exp[i][j][0] = Exp[j][j][0] + exp_weight
            Exp[i][i] = min(Exp[i])
        i = len(L)
        ckpt[L[i - 1]] = Exp[i][i][1:4]
        while i != 0:
            sub_list = Exp[i]
            j = sub_list.index(min(sub_list))
            if j > 0:
                ckpt[L[j - 1]] = Exp[j][j][1:4]
            i = j
        return ckpt