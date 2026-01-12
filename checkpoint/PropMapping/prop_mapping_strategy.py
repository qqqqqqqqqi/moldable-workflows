import csv

from Base import util
from Base.decompose import *


class PropMappingStrategy:

    def __init__(self, w, G, P, p_fail, lam, gamma, eta, ccr):
        self.w = w
        self.G = G
        self.P = P
        self.p_fail = p_fail
        self.lam = lam
        self.gamma = gamma
        self.eta = eta
        self.ccr = ccr
        self.assign = {}
        self.schedule = [[] for _ in range(self.P)]
        self.alloc = {}
        self.avail = {i: 0.0 for i in range(self.P)}
        self.ckpt = {node: [0, 0, 0] for node in self.G.nodes}
        self.FJB={}
        self.chainOnP=[[] for i  in range(0,self.P)]

    def decompose(self):
        G=self.G
        if self.w == 'Epigenomics':
            self.FJB = decompose_epigenomics(G)
        elif self.w == 'Seismology':
            self.FJB = decompose_seismology(G)
        elif self.w == 'Soykb':
            self.FJB = decompose_soykb(G)
        elif self.w == 'Srasearch':
            self.FJB = decompose_srasearch(G)
        elif self.w == 'Genome':
            self.FJB = decompose_genome(G)
        elif self.w == 'Cycle':
            self.FJB = decompose_cycle(G)
        elif self.w == 'Blast':
            self.FJB = decompose_blast(G)
        elif self.w == 'Bwa':
            self.FJB = decompose_bwa(G)
        elif self.w == 'Montage':
            self.FJB = decompose_montage(G)

    def get_schedule(self,outdir):
        pass


    def  checkpoint(self):
        pass

    def onProcessor(self,subG, p_num,p_set):
        L = []
        weak_components = nx.weakly_connected_components(subG)
        graphs = [subG.subgraph(c) for c in weak_components]
        for g in graphs:
            linear_chain = list(nx.topological_sort(g))
            L.append(linear_chain)
            for node in linear_chain:
                self.assign[node] = p_num
                self.alloc[node]=p_set
                self.G.nodes[node]['d_weight'] = util.amdahl(self.G.nodes[node]['weight'], self.G.nodes[node]['alpha'], p_num)
        return L

    def Map(self,G, p):
        if nx.number_of_nodes(G) == 0:
            return
        if p[0] == p[1]:
            L = self.onProcessor(G, 1,[i for i in range(p[0],p[1]+1)])
            self.schedule[p[0]].extend(sum(L,[]))
            self.chainOnP[p[0]]+=L
            return

        root_list = []
        for node in G.nodes():
            if len(G.pred[node]) == 0:
                root_list.append(node)
        # print root_list

        if len(root_list) == 1:  # 找链
            pointer = root_list[0]
            nbunch = [pointer]
            while G.out_degree(pointer) == 1:
                pointer = list(G.successors(pointer))[0]
                nbunch.append(pointer)
            chain = G.subgraph(nbunch)
            L = self.onProcessor(chain, p[1] - p[0] + 1,[i for i in range(p[0],p[1]+1)])
            self.chainOnP[p[0]]+=L
            for i in range(p[0], p[1] + 1):
                self.schedule[i].extend(sum(L,[]))

            if G.out_degree(pointer) == 0:
                return
            else:
                nbunch_h = [node for node in G.nodes() if node not in nbunch]
                H = G.subgraph(nbunch_h)
                self.Map(H, p)
                return

        cc = nx.number_weakly_connected_components(G)
        if cc == 1:
            root_list.sort()
            nbunch = self.FJB[tuple(root_list)]
            nbunch_h = [node for node in G.nodes() if node not in nbunch]
            g1 = G.subgraph(nbunch_h)
            g2 = G.subgraph(nbunch)
            self.Map(g1, p)
            self.Map(g2, p)
            return
        else:
            subG, subP = self.propMap(G, p)
            start = p[0]
            for i in range(0, len(subP)):
                end = start + subP[i] - 1
                self.Map(subG[i], [start, end])
                start = end + 1
        return


    def propMap(self,G, p):
        rho = p[1] - p[0] + 1
        n = nx.number_weakly_connected_components(G)
        k = min(rho, n)
        subG = [nx.DiGraph()] * k
        subP = [1] * k
        W = [0] * k
        weightG = {}
        components = nx.weakly_connected_components(G)
        graphs = [G.subgraph(c) for c in components]
        for g in graphs:
            weightG[g] = compute_subgraph_weight(g)

        if n >= rho:
            while (len(weightG) != 0):
                i = max(weightG, key=weightG.get)
                j = W.index(min(W))
                W[j] += weightG[i]
                subG[j] = nx.compose(subG[j], i)
                del weightG[i]
        else:
            i = 0
            for key in weightG.keys():
                subG[i] = key
                W[i] = weightG[key]
                i += 1
            rho -= n
            while rho != 0:
                j = W.index(max(W))
                subP[j] += 1
                W[j] /= subP[j]
                rho -= 1
        return subG, subP

    def output2csv(self, filename):
        with open(filename, 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['task_id', 'weight', 'proc', 'ckpt', 'C', 'V','nb'])
            for node in self.G.nodes:
                w.writerow(
                    [node, self.G.nodes[node]['d_weight'], "_".join(map(str, self.alloc[node])),
                     self.ckpt[node][0], self.ckpt[node][1],
                     self.ckpt[node][2],len(self.alloc[node])])
            for i in range(len(self.schedule)):
                w.writerow([i] + self.schedule[i])
            w.writerow(['makespan', 'lambda'])
            w.writerow([self.get_makespan(), self.lam])

    def get_makespan(self):
        pass