from strategy import Strategy
import util


class CkptAll(Strategy):
    def __init__(self, w, G, P, p_fail, lam, ccc, vvv,ccr):
        super().__init__(w, G, P, p_fail, lam, ccc, vvv,ccr)
        self.mapping_order = []

    def allocation(self):
        visited = {}
        A = {}
        alloc = {}
        for node in self.G.nodes:
            self.assign[node] = 1
            visited[node] = 0
            A[node] = [i for i in range(1, self.P + 1)]
            ko = 0
            size = self.G.nodes[node]['weight']
            alpha = self.G.nodes[node]['alpha']
            self.G.nodes[node]['d_weight'] = util.amdahl(size, alpha, self.assign[node])
            alloc[node] = []
            for k in range(1, len(A[node])):
                pj_ko = util.amdahl(size, alpha, A[node][ko])
                pj_k = util.amdahl(size, alpha, A[node][k])
                benefit = util.compute_benefit(pj_ko, pj_k, A[node][ko], A[node][k])
                rel_improvement = util.compute_rel_runtime_improvement(pj_ko, pj_k)
                if benefit > 0 and rel_improvement >= util.THRESHOLD:
                    ko = k
                    alloc[node].append([A[node][k], benefit])

        L, v_cp = util.compute_L(self.G)
        W = util.compute_W(self.G, self.assign)
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
                self.G.nodes[vb]['d_weight'] = util.amdahl(self.G.nodes[vb]['weight'], self.G.nodes[vb]['alpha'], ab)
                alloc[vb].pop(0)
                visited[vb] = 1
                L, v_cp = util.compute_L(self.G)
                W = util.compute_W(self.G, self.assign)
            else:
                break

    def shrink_mapping(self):
        ticks = []
        bl = util.compute_bottom_weight(self.G)[0]
        current = 0.0
        scheduleNb = self.G.number_of_nodes()
        parents = util.getParentsNb(self.G)

        ft = {}  # the finish time of each task
        stime = {node: 0.0 for node in self.G.nodes}
        while (scheduleNb > 0):
            for a in self.avail:
                self.avail[a] = max(self.avail[a], current)
            ready = [(i, bl[i]) for i in parents if parents[i] == 0 and stime[i] <= current]
            order_ready = sorted(ready, key=lambda x: x[1], reverse=True)
            if len(order_ready) > 0:
                for node in order_ready:
                    rn = node[0]
                    self.mapping_order.append(rn)
                    origin_end_time = super().find_slots(self.assign[rn], stime[rn], self.G.nodes[rn]['d_weight'])
                    np = self.find_better_allocation(self.G.nodes[rn], self.assign[rn], stime[rn], origin_end_time)
                    self.G.nodes[rn]['d_weight'] = util.amdahl(self.G.nodes[rn]['weight'], self.G.nodes[rn]['alpha'],
                                                               np)
                    self.assign[rn] = np
                    etime = self.find_slots_alloc(rn, self.assign[rn], stime[rn], self.G.nodes[rn]['d_weight'])
                    ft[rn] = etime
                    for nd in self.G.successors(rn):
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
                exec_time = util.amdahl(node['weight'], node['alpha'], p)
                end_time = exec_time + last_start_time
                if (end_time < origin_end_time):
                    better_np = p
            current_start_time = last_start_time
        return better_np

    def checkpoint(self):
        for node in self.G.nodes:
            fs = {node: self.G.nodes[node]['ck_ext']}
            for succ in self.G.successors(node):
                fs.update(self.G.edges[node,succ]['filelist'])
            self.ckpt[node][0] = 1
            self.ckpt[node][1] = sum(fs.values()) * self.ccc
            self.ckpt[node][2] = sum(fs.values()) * self.vvv

    def get_schedule(self, outdir):
        self.allocation()
        self.shrink_mapping()
        self.checkpoint()
        filename = f"{outdir}/0_{self.w}_{self.P}_{self.p_fail}_{self.ccr:.1e}.csv"
        self.output2csv(filename)

    def get_makespan(self):
        ft = {}
        avail = [0 for _ in range(len(self.schedule))]
        for node in self.mapping_order:
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

