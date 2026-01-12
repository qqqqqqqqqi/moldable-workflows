import csv
import sys
import time

from Base import util
from Base.strategy import Strategy
from Solver.algorithm_Chen13 import *
from Solver.mixtesim import *


class CkptAll(Strategy):
    def __init__(self, w, G, P, p_fail, lam, gamma, eta, ccr):
        super().__init__(w, G, P, p_fail, lam, gamma, eta, ccr)
        self.mapping_order = []

    def allocation(self):
        dag_parser = DagParser()
        dag = dag_parser.parse(self.G)

        grid_parser = MixteSimPlatformParser()
        grid = grid_parser.parse(self.P)
        stats = {
            "lp_write_start": 0,
            "lp_write_end": 0,
            "lp_size_bytes": 0
        }
        compNodeId2Id = {}
        fakeId2compNode = {}
        nodeId = 1
        for node in dag.get_computational_nodes():
            compNodeId2Id[node.id] = nodeId
            fakeId2compNode[nodeId] = node
            nodeId += 1
        lpcreator = LpCreatorChen13(compNodeId2Id, fakeId2compNode)
        solver = CplexGlpkSolverChen13()
        producer = Chen13Producer(compNodeId2Id, fakeId2compNode)
        stats["lp_write_start"] = time.time()
        prog_str = lpcreator.create_linear_program(dag, grid)
        out = sys.stdout
        ofile = f'{self.w}_{self.P}.mod'
        if ofile is not None:
            out = open(ofile, "w")

        out.write(prog_str)

        if ofile is not None:
            out.close()

        lp_out_fname = ofile.replace(".mod", ".sol")

        solver.solve(ofile, lp_out_fname)

        res_hash = solver.read_result(lp_out_fname)

        solution = producer.get_solution(res_hash, dag, grid, None)
        stats["lp_write_end"] = time.time()

        out = sys.stdout

        for jobid in solution.keys():
            print(f"@SOLUTION[task,psize]: {jobid} {solution[jobid][0]}", file=out)
            nd = str(jobid)
            self.assign[nd] = int(solution[jobid][0])
            self.G.nodes[nd]['d_weight'] = util.amdahl(self.G.nodes[nd]['weight'], self.G.nodes[nd]['alpha'],
                                                       self.assign[nd])

        with open("runtime.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ckptAll", self.w, self.P, stats["lp_write_end"] - stats["lp_write_start"]])

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
                fs.update(self.G.edges[node, succ]['filelist'])
            self.ckpt[node][0] = 1
            self.ckpt[node][1] = sum(fs.values()) * self.gamma
            self.ckpt[node][2] = sum(fs.values()) * self.eta

    def get_schedule(self, outdir):
        start = time.perf_counter()
        self.allocation()
        self.shrink_mapping()
        self.checkpoint()
        end = time.perf_counter()
        filename = f"{outdir}/0_{self.w}_{self.P}_{self.p_fail}_{self.ccr:.1e}.csv"
        self.output2csv(filename)
        return end - start, self.get_makespan()

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

