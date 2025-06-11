import csv


class Strategy:

    def __init__(self, w, G, P, p_fail, lam, ccc, vvv,ccr):
        self.w = w
        self.G = G
        self.P = P
        self.p_fail = p_fail
        self.lam = lam
        self.ccc = ccc
        self.vvv = vvv
        self.ccr = ccr
        self.assign = {}
        self.schedule = [[] for _ in range(self.P)]
        self.alloc = {}
        self.avail = {i: 0.0 for i in range(self.P)}
        self.ckpt = {node: [0, 0, 0] for node in self.G.nodes}

    def allocation(self):
        pass

    def shrink_mapping(self):
        pass

    def checkpoint(self):
        pass

    def find_better_allocation(self, node, oP, min_start_time, origin_end_time):
        pass

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

    def find_slots(self, assign_p, max_pred, runtime):
        sort_avail = sorted(self.avail.items(), key=lambda x: x[1])
        minStart = max(max_pred, list(sort_avail)[assign_p - 1][1])
        return minStart + runtime

    def output2csv(self, filename):
        with open(filename, 'w') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(['parent', 'child','filelist'])
            for edge in self.G.edges(data=True):
                fs=edge[2]['filelist']
                fl=[]
                for k,v in fs.items():
                    fl+=[k,v*self.ccc]
                w.writerow([edge[0], edge[1]]+fl)  # weight:total weight
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

    def get_schedule(self, outdir):
        pass

    def get_makespan(self):
        return 0
