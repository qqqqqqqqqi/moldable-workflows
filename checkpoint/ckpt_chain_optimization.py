import math
from math import inf

import networkx as nx
from strategy import Strategy
import util

class CkptChainOptimization(Strategy):
    def __init__(self, w, G, P, p_fail, lam, ccc, vvv,ccr):
        super().__init__(w, G, P, p_fail, lam, ccc, vvv,ccr)
        self.mG = nx.DiGraph()
        self.chain2node = {}  # [chain:node]
        self.node2chain = {}  # [node: chain]
        self.mapping_order = []

    def allocation(self):
        visited = {}
        A = {}
        alloc = {}

        for node in self.mG.nodes:
            self.assign[node] = 1
            visited[node] = 0
            A[node] = [i for i in range(1, self.P + 1)]
            ko = 0
            size = self.mG.nodes[node]['weight']
            alpha = self.mG.nodes[node]['alpha']
            self.mG.nodes[node]['d_weight'] = util.amdahlM(size, alpha, self.assign[node])

            alloc[node] = []
            for k in range(1, len(A[node])):
                pj_ko = util.amdahlM(size, alpha, A[node][ko])
                pj_k = util.amdahlM(size, alpha, A[node][k])
                benefit = util.compute_benefit(pj_ko, pj_k, A[node][ko], A[node][k])
                rel_improvement = util.compute_rel_runtime_improvement(pj_ko, pj_k)
                if benefit > 0 and rel_improvement >= util.THRESHOLD:
                    ko = k
                    alloc[node].append([A[node][k], benefit])

        L, v_cp = util.compute_L(self.mG)
        W = util.compute_W(self.mG, self.assign)
        tl = util.compute_pred_level(self.G)[0]
        while L > W / self.P:
            vb, ab, bb = None, self.P, 0.0
            md = util.computeMd(tl, visited, self.assign)
            for node in v_cp:
                if len(alloc[node]) == 0: continue
                bf, a = alloc[node][0][1], alloc[node][0][0]
                s = a - self.assign[node]
                if md[tl[node]] + s <= self.P and bf > bb:
                    vb, ab, bb = node, a, bf
            if vb is not None:
                self.assign[vb] = ab
                self.mG.nodes[vb]['d_weight'] = util.amdahlM(self.mG.nodes[vb]['weight'], self.mG.nodes[vb]['alpha'],
                                                             ab)
                alloc[vb].pop(0)
                visited[vb] = 1
                L, v_cp = util.compute_L(self.mG)
                W = util.compute_W(self.mG, self.assign)
            else:
                break

    def shrink_mapping(self):
        ticks = []
        bl = util.compute_bottom_weight(self.mG)[0]
        current = 0.0
        scheduleNb = self.mG.number_of_nodes()
        parents = util.getParentsNb(self.mG)
        ft = {}  # the finish time of each task
        stime = {node: 0.0 for node in self.mG.nodes}

        while (scheduleNb > 0):
            for a in self.avail:
                self.avail[a] = max(self.avail[a], current)
            ready = [(i, bl[i]) for i in parents if parents[i] == 0 and stime[i] <= current]
            order_ready = sorted(ready, key=lambda x: x[1], reverse=True)
            if len(order_ready) > 0:
                for node in order_ready:
                    rn = node[0]
                    self.mapping_order.append(rn)
                    origin_end_time = self.find_slots(self.assign[rn], stime[rn], self.mG.nodes[rn]['d_weight'])
                    np = self.find_better_allocation(self.mG.nodes[rn], self.assign[rn], stime[rn], origin_end_time)
                    self.mG.nodes[rn]['d_weight'] = util.amdahlM(self.mG.nodes[rn]['weight'],
                                                                 self.mG.nodes[rn]['alpha'],
                                                                 np)
                    self.assign[rn] = np
                    etime = self.find_slots_alloc(rn, self.assign[rn], stime[rn], self.mG.nodes[rn]['d_weight'])
                    ft[rn] = etime
                    for nd in self.mG.successors(rn):
                        parents[nd] -= 1
                        stime[nd] = max(stime[nd], ft[rn])
                    scheduleNb -= 1
                    parents.pop(rn)
                    ticks.append(ft[rn])
            else:
                ticks.sort()
                current = ticks.pop(0)

    def find_better_allocation(self, node, oP, min_start_time, origin_end_time):
        sort_avail = list(sorted(self.avail.items(), key=lambda x: x[1]))
        origin_start_time = max(min_start_time, sort_avail[oP - 1][1])
        current_start_time = origin_start_time
        better_np = oP
        for p in range(oP - 1, 0, -1):
            last_proc_time = sort_avail[p - 1][1]
            last_start_time = max(min_start_time, last_proc_time)
            if last_start_time < current_start_time:
                exec_time = util.amdahlM(node['weight'], node['alpha'], p)
                end_time = exec_time + last_start_time
                if (end_time < origin_end_time):
                    better_np = p
            current_start_time = last_start_time
        return better_np

    def merge_chain2node(self):
        nodes = list(nx.topological_sort(self.G))
        nodeSucc = {}
        while len(nodes) > 0:
            pointer = nodes[0]
            weights = [self.G.nodes[pointer]['weight']]
            alphas = [self.G.nodes[pointer]['alpha']]
            merged = [pointer]
            id = pointer
            while self.G.out_degree(pointer) == 1:
                prec = pointer
                pointer = list(self.G.successors(pointer))[0]
                if (self.G.in_degree(pointer) == 1):
                    weights.append(self.G.nodes[pointer]['weight'])
                    alphas.append(self.G.nodes[pointer]['alpha'])
                    merged.append(pointer)
                    self.chain2node[prec] = id
                    nodes.remove(prec)
                else:
                    pointer = prec
                    break
            self.mG.add_node(id, weight=weights, alpha=alphas)
            self.chain2node[pointer] = id
            nodes.remove(pointer)
            nodeSucc[id] = list(self.G.successors(pointer))
            self.node2chain[id] = merged
        for node in nodeSucc:
            for succ in nodeSucc[node]:
                self.mG.add_edge(node, succ)

    def decompose(self):
        for node in self.G.nodes:
            self.alloc[node] = self.alloc[self.chain2node[node]]
            self.assign[node] = self.assign[self.chain2node[node]]
            self.G.nodes[node]['d_weight'] = util.amdahl(self.G.nodes[node]['weight'], self.G.nodes[node]['alpha'],
                                                         self.assign[node])
        for i in range(len(self.schedule)):
            ns = []
            for k, node in enumerate(self.schedule[i]):
                ns.extend(self.node2chain[node])
            self.schedule[i] = ns

   # checkpoint tasks with crossover dependences / checkpoint chain by DP
    def checkpoint(self):
        tmpCkpt = []  # tasks to be checkpointed
        noCkpt = []
        for node in self.mG.nodes:
            L = self.node2chain[node]
            if len(L) > 1:
                self.ckpt.update(self.ckpt_chain(L, self.assign[L[0]]))
            else:
                tag = False
                if self.G.out_degree(node) == 0:
                    tag = True
                    tmpCkpt.append(node)
                else:
                    for succ in self.mG.successors(node):
                        if self.alloc[succ] != self.alloc[node]:
                            tmpCkpt.append(node)
                            tag = True
                            break
                        else:
                            pros = self.alloc[node]
                            for pro in pros:
                                i1 = self.schedule[pro].index(node)
                                if self.schedule[pro][i1 + 1] != succ:
                                    tag = True
                                    tmpCkpt.append(node)
                                    break
                        if tag:
                            break
                if not tag:
                    noCkpt.append(node)

        for node in tmpCkpt:
            fs = {}
            p = self.alloc[node][0]
            curr = self.schedule[p].index(node)
            lastCkpt = curr - 1
            while lastCkpt >= 0 and self.schedule[p][lastCkpt] in noCkpt:
                lastCkpt -= 1

            for i in range(lastCkpt + 1, curr + 1):
                fs.update({self.schedule[p][i]: self.G.nodes[self.schedule[p][i]]['ck_ext']})
                for succ in self.G.successors(self.schedule[p][i]):
                    if succ not in self.schedule[p][lastCkpt + 1:curr + 1]:
                        fs.update(self.G.edges[self.schedule[p][i], succ]['filelist'])
            self.ckpt[node][0] = 1
            self.ckpt[node][1] = sum(fs.values()) * self.ccc
            self.ckpt[node][2] = sum(fs.values()) * self.vvv

    def ckpt_chain(self, L, assign_p):  # checkpoint linear chain
        Exp = [[[inf for k in range(4)] for i in range(0, len(L) + 1)] for j in range(0, len(L) + 1)]
        Exp[0][0] = [0, 1, 0, 0]
        ckpt = {}
        for i in range(1, len(L) + 1):
            for j in range(0, i):
                w, rv = util.T(j, i, L, self.G)
                c = util.C(j, i, L, self.G)
                v = c
                c *= self.ccc
                v *= self.vvv
                rv *= self.ccc
                exp_weight = math.exp(self.lam * assign_p * w) * (w + v + rv) + c - rv
                Exp[i][j][1] = 1
                Exp[i][j][2] = c
                Exp[i][j][3] = v
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

    def reduce(self):
        order = []
        for node in self.mapping_order:
            order.extend(self.node2chain[node])
        visited = []
        new_schedule = [[] for _ in range(self.P)]
        lasttime = [0 for _ in range(self.P)]
        ft = {}
        for node in order:
            if node not in visited:
                L = [node]
                proc = self.alloc[node]
                if self.ckpt[node][0] == 0:
                    index = self.schedule[proc[0]].index(node) + 1
                    while self.ckpt[self.schedule[proc[0]][index]][0] == 0:
                        L.append(self.schedule[proc[0]][index])
                        index += 1
                    L.append(self.schedule[proc[0]][index])
                visited.extend(L)
                op = self.find_opt2ckpt(L,self.assign[node])
                sub = self.assign[node] - op
                minstart = 0
                if self.G.in_degree(node) > 0:
                    minstart = max([ft[nd] for nd in self.G.predecessors(node)])
                if sub > 0:
                    sp = sorted([[i, lasttime[i]] for i in proc], key=lambda x: x[1])[:op]
                    newP = [d[0] for d in sp]
                    for nd in L:
                        self.assign[nd] = op
                        self.alloc[nd] = newP
                        runtime = util.amdahl(self.G.nodes[nd]['weight'],
                                              self.G.nodes[nd]['alpha'], op)
                        self.G.nodes[nd]['d_weight'] = runtime
                        ft[nd] = minstart + runtime
                        minstart += runtime
                    for p in newP:
                        new_schedule[p].extend(L)
                        lasttime[p] = minstart
                else:
                    minstart = max(minstart, max([lasttime[p] for p in proc]))
                    for nd in L:
                        runtime = self.G.nodes[nd]['d_weight']
                        ft[nd] = minstart + runtime
                        minstart += runtime
                    for p in proc:
                        new_schedule[p].extend(L)
                        lasttime[p] = minstart
        self.schedule = new_schedule

    def find_opt2ckpt(self, L,max_p):
        w = [self.G.nodes[node]['weight'] for node in L]
        alpha = [self.G.nodes[node]['alpha'] for node in L]
        bestP = 1
        f = util.C(0, len(L), L, self.G)
        V = f * self.vvv
        C = f * self.ccc
        R = util.R(0, len(L), L, self.G) * self.ccc
        runtime = util.amdahlM(w, alpha, 1)
        minE = math.exp(self.lam * runtime) * (runtime + V + R) + C - R
        for i in range(2, max_p+1):
            nw = util.amdahlM(w, alpha, i)
            E = math.exp(self.lam * i * nw) * (nw + V + R) + C - R
            if E < minE:
                minE = E
                bestP = i
        return bestP

    def find_slots_alloc(self, node, assign_p, max_pred, runtime):
        sort_avail = sorted(self.avail.items(), key=lambda x: x[1])
        minStart = max(max_pred, list(sort_avail)[assign_p - 1][1])
        alloc = []
        for k in dict(sort_avail[:assign_p]):
            self.avail[k] = minStart + runtime
            self.schedule[k].append(node)
            alloc.append(k)
        self.alloc[node] = alloc
        return minStart + runtime

    def get_schedule(self, outdir):
        self.merge_chain2node()
        self.allocation()
        self.shrink_mapping()
        self.decompose()
        self.checkpoint()
        self.reduce()
        filename = f"{outdir}/4_{self.w}_{self.P}_{self.p_fail}_{self.ccr:.1e}.csv"
        self.output2csv(filename)

    def get_makespan(self):
        ft = {}
        mp_nodes=[]
        for node in self.mapping_order:
            mp_nodes.extend(self.node2chain[node])
        avail = [0 for _ in range(len(self.schedule))]
        for node in mp_nodes:
            proc = self.alloc[node]
            max_pred = 0
            for pred in self.G.predecessors(node):
                if max_pred < ft[pred]:
                    max_pred = ft[pred]
            earliest_p = max(avail[i] for i in proc)
            min_start = max(earliest_p, max_pred)
            runtime = self.G.nodes[node]['d_weight'] + sum(self.ckpt[node][1:3])
            for k in proc:
                avail[k] = min_start + runtime
            ft[node] = min_start + runtime
        return max(ft.values())
