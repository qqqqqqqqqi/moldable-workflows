# Moldable Task Scheduling Package
# Copyright (C) 2013 Sascha Hunold
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import math
import re

from lp_base import *
from node_time import get_time_of_node
import scheduler_config as conf

class VirtualTasks:

    def __init__(self, id):
        self.id = id
        self.xlist = []
        self.wlist = []
        self.nb_vtasks = 0

    def get_id(self):
        return self.id

    def append_execution_time(self, x):
        self.xlist.append(x)

    def append_work(self, w):
        self.wlist.append(w)

    def set_nb_vtasks(self, nb_vtasks):
        self.nb_vtasks = nb_vtasks

    def get_nb_vtasks(self):
        return self.nb_vtasks

    def get_execution_time_at_idx(self, idx):
        return self.xlist[idx]

    def get_work_at_idx(self, idx):
        return self.wlist[idx]

    def get_nb_of_work_items(self):
        return len(self.wlist)

class CVTA:

    def __init__(self, options=None):
        self.options = options

    # returns hash node.id -> VirtualTasks
    def create_virtual_tasks(self, dag, grid):
        res_hash = {}

        m = grid.get_p()
        for node in dag.get_computational_nodes():
            mj = 0
            s = 1

            vtasks = VirtualTasks(node.id)

            for i in range(1,m+1):
                p1 = get_time_of_node(node, i, grid)

                if i < m:
                    p2 = get_time_of_node(node, i+1, grid)
                else:
                    p2 = p1

                if (i == m) or (p1 != p2):
                    mj += 1

                    vtasks.append_execution_time(p1)

                    if i != m:
                        ws =  get_time_of_node(node, s, grid) * s
                        wi1 = p2 * (i+1)
                        # if self.options is not None and self.options.verbose:
                        # print( "node.id", node.id, "i:", i, " ws:", ws, " wi1:", wi1, " diff:", (wi1-ws))
                        vtasks.append_work(wi1 - ws)
                        s = i+1

            vtasks.set_nb_vtasks(mj)
            res_hash[node.id] = vtasks


        return res_hash




class LpCreatorChen13:

    def __init__(self, compNodeId2Id, fakeId2compNode, options=None):
        self.compNodeId2Id = compNodeId2Id
        self.fakeId2compNode = fakeId2compNode
        self.options = options

    def create_linear_program(self, dag, grid):

        cvta = CVTA(self.options)
        vtask_hash = cvta.create_virtual_tasks(dag, grid)

        NP = grid.get_p()
        vs = VarStore()

        buf = WriteBuffer()
        buf.write("minimize MAKESPAN: Cmax;\n")
        vs.add("Cmax")

        buf.write_constraint_with_option("0 <= C[i];\n", "{i in N}")
        buf.write_constraint_with_option("C[i] <= L;\n", "{i in N}")
        buf.write_constraint_with_option("x[i] <= p[i,1];\n", "{i in N}")
        buf.write_constraint_with_option("x[i] <= C[i];\n", "{i in N}")

        for node in dag.get_computational_nodes():
            nodeId = self.compNodeId2Id[node.id]

            for child in dag.get_computational_children(node.id):
                childId = self.compNodeId2Id[child.id]
                buf.writec("C[%s] + x[%s] <= C[%s];\n" % ( nodeId, childId, childId ))

            buf.write_constraint_with_option("xs_%s[i] <= x[%s];\n" % ( nodeId, nodeId ), "{i in M%s}" % ( nodeId ) )

            buf.write_constraint_with_option("0 <= xs_%s[i];\n" % ( nodeId ), "{i in M%s_m1}" % ( nodeId ) )
            buf.write_constraint_with_option("xs_%s[i] <= vx_%s[i];\n" % ( nodeId, nodeId ), "{i in M%s_m1}" % ( nodeId ) )

            vtasks = vtask_hash.get(node.id)
            last_m_idx = vtasks.get_nb_vtasks()

            buf.writec("xs_%s[%s] = vx_%s[%s];\n" % ( nodeId, last_m_idx, nodeId, last_m_idx ))

            buf.write_constraint_with_option("xs_%s[i+1] <= xs_%s[i];\n" % ( nodeId, nodeId ), "{i in M%s_m1}" % ( nodeId ) )

            buf.write_constraint_with_option("xs_%s[i] - xs_%s[i+1] <= vx_%s[i] - vx_%s[i+1];\n" % ( nodeId, nodeId, nodeId, nodeId ), "{i in M%s_m1}" % ( nodeId ) )

            for i in range(0, vtasks.get_nb_vtasks()-2):

                timej  = vtasks.get_execution_time_at_idx(i)
                timej1 = vtasks.get_execution_time_at_idx(i+1)
                timej2 = vtasks.get_execution_time_at_idx(i+2)

                buf.writec("(xs_%s[%s] - xs_%s[%s])/(%s - %s) <= (xs_%s[%s] - xs_%s[%s])/(%s - %s);\n" %\
                            ( nodeId, i+1, nodeId, i+2, timej, timej1, nodeId, i+2, nodeId, i+3, timej1, timej2 ) )

            # work constraints
            buf.writec("wq_%s[%s] = 0;\n" % (nodeId, vtasks.get_nb_vtasks()))
            for i in range(0, vtasks.get_nb_of_work_items()):
                vworki = vtasks.get_work_at_idx(i)
                timei  = vtasks.get_execution_time_at_idx(i)
                timei1 = vtasks.get_execution_time_at_idx(i+1)

                buf.writec("wq_%s[%s] = %s * ( 1 - (xs_%s[%s] - xs_%s[%s])/(%s - %s) );\n" % \
                           (nodeId, i+1, vworki, nodeId, i+1, nodeId, i+2, timei, timei1) )


            buf.writec("wtot[%s] = sum{i in M%s} wq_%s[i];\n" % (nodeId, nodeId, nodeId) )


        buf.writec("( sum { j in N } wtot[j] ) + P <= W;\n")

        buf.writec("L <= Cmax;\n")
        buf.writec("W/m <= Cmax;\n")


        vs.add("C{N} >= 0")
        #vs.add("xs{N,M} >= 0")
        vs.add("x{N} >= 0")
        vs.add("L >= 0")
        vs.add("W >= 0")
        vs.add("wtot{N} >= 0")  # w^_j(x_j)

        # buf.writec("sum {j in N} sum {l in M} xs[j,l] * (l * p[j,l]) <= m * Cmax;\n")
        #
        # buf.write_constraint_with_option("sum {l in M} xs[j,l] = 1;\n", "{j in N}")

        buf2 = WriteBuffer()  # buffer for lp header

        buf2.write("param m integer, >= 0;  /* nr of processors */ \n")
        buf2.write("param n integer, >= 0;  /* nb jobs */\n")
        buf2.write("set N := 1..n;\n")
        buf2.write("set M := 1..m;\n")
        buf2.write("param p{i in N, j in M};\n")
        buf2.write("param mjob{i in N};\n");
        buf2.write("param P;\n")


        for node in dag.get_computational_nodes():
            nodeId = self.compNodeId2Id[node.id]
            vtasks = vtask_hash.get(node.id)
            buf2.write("set M%s := 1..%s;\n" % (nodeId, vtasks.get_nb_vtasks()))
            buf2.write("set M%s_m1 := 1..%s;\n" % (nodeId, vtasks.get_nb_vtasks()-1))
            buf2.write("set M%s_m2 := 1..%s;\n" % (nodeId, vtasks.get_nb_vtasks()-2))
            buf2.write("param vx_%s{ M%s };\n" % (nodeId, nodeId) )
            # w-
            vs.add("wq_%s{ M%s } >= 0" % (nodeId, nodeId) )
            vs.add("xs_%s{ M%s } >= 0" % (nodeId, nodeId) )

        for var in vs.get_variables():
            buf2.write("var %s;\n" % (var))

        # okay back to buf1 again

        buf.write("data;\n")

        N = len(dag.get_computational_nodes())

        buf.write("param m := %d;\n" % (NP) )
        buf.write("param n := %d;\n" % ( N ) )

        buf.write("param p : ")
        for p in range(1, NP+1):
            buf.write("%d " % (p))
        buf.write(":= \n")
        for fakeId in range(1, N+1):
            compNode = self.fakeId2compNode[fakeId]
            buf.write("%s " % (fakeId))
            for p in range(1, NP+1):
                buf.write("%g " % (get_time_of_node(compNode, p, grid)))
            buf.write("\n")
        buf.write(";\n")

        buf.write("param mjob := ")
        for node in dag.get_computational_nodes():
            nodeId = self.compNodeId2Id[node.id]
            mj = vtask_hash.get(node.id).get_nb_vtasks()
            buf.write("[%d] %d " % (nodeId, mj))
        buf.write(";\n")

        total = 0
        for node in dag.get_computational_nodes():
            total += get_time_of_node(node, 1, grid)
        buf.write("param P := %s;\n" % ( total ))


        # fill virtual execution time array vx_j[i]
        for node in dag.get_computational_nodes():
            nodeId = self.compNodeId2Id[node.id]
            vtasks = vtask_hash.get(node.id)
            buf.write("param vx_%s :=" % (nodeId))
            for i in range(0, vtasks.get_nb_vtasks()):
                buf.write(" [%s] %g" % ( (i+1), vtasks.get_execution_time_at_idx(i)))
            buf.write(";\n")


        buf.write("end;\n")

        return repr(buf2) + repr(buf)




class GlpkSolverChen13(LinProgSolver):

    def solve(self, lp_fname, sol_fname):
        log_fname = sol_fname.replace(".sol", ".log")
        call = "%s -m %s  -o %s --log %s > /dev/null" % (conf.GLPSOL_PATH, lp_fname, sol_fname, log_fname)
        #call = "glpsol -m %s  -o %s --log %s > /dev/null" % (lp_fname, sol_fname, log_fname)
        os.system(call)


    def read_result(self, sol_fname):
        # returns a hash { xs[i,j] -> runtime }
        fh = open(sol_fname)
        res_content = fh.readlines()
        fh.close()

        res_hash = {}
        for line in res_content:
            vals = line.split()

            # debugging
            # if len(vals) >= 1 and vals[1].startswith("x["):
            #     print line,

            if len(vals) >= 4 and vals[1].startswith("xs_"):
                val = float(vals[3])
                if val != 0.0:
                    res_hash[ vals[1] ] = val

        return res_hash


class CplexGlpkSolverChen13(LinProgSolver):

    # we take a GLPK program and convert it to CPLEX
    # with GLPSOL
    def solve(self, lp_fname, sol_fname):

        cplex_in_fname = lp_fname.replace(".mod", ".lp")

        call = "%s --check --math %s --wlp %s > /dev/null" % ( conf.GLPSOL_PATH, lp_fname, cplex_in_fname )
        os.system(call)

        cplex_cmd = """
read %s lp
primopt
display solution objective
display solution variables -
quit
""" % ( cplex_in_fname )

#optimize

        #print cplex_cmd

        call = "echo \"%s\" | %s > %s" % (cplex_cmd, conf.CPLEX_PATH,sol_fname)
        os.system(call)

        #remove cplex lp file
        os.unlink(cplex_in_fname)


    def read_result(self, sol_fname):
        # returns a hash [ x[i] -> runtime ]
        fh = open(sol_fname)
        res_content = fh.readlines()
        fh.close()

        res_hash = {}
        for line in res_content:
            vals = line.split()
            if len(vals) == 2 and re.match("^xs\_(\d+)\(\d+\)", vals[0]):
                # change x(1) -> x[1]
                xs_val = vals[0]
                xs_val = xs_val.replace("(", "[")
                xs_val = xs_val.replace(")", "]")
                #print xs_val
                res_hash[ xs_val ] = vals[1]

        return res_hash


#
# rounding solution by Chen and Chu (2013)
#
class Chen13Producer(LpSolutionProducer):

    def __init__(self, compNodeId2Id, fakeId2compNode):
        self.compNodeId2Id = compNodeId2Id
        self.fakeId2compNode = fakeId2compNode

    def compute_mu(self, rho, m):
        mu = m*(2+rho) - math.sqrt( (2+2*rho+rho*rho)*m*m - 2*m*(1+rho))
        mu /= 2
        mu = int( mu )
        return mu

    def get_solution(self, res_hash, dag, grid, options):
        solution = {}

#        RHO = 0.99
#        MU  = self.compute_mu(RHO, grid.get_p())

        m = grid.get_p()
        RHO = 0
        MU  = m - math.sqrt(2*m*m - 2*m)/2

        # if options.verbose:
        #     print ("RHO:", RHO)
        #     print ("MU :", MU)
        cvta = CVTA()
        vtask_hash = cvta.create_virtual_tasks(dag, grid)

        # convert res_hash to  x [ jobid ] [ proc ] = val
        x = {}
        pat1 = re.compile("xs\_(\d+)\[(\d+)\]")
        res_var_list = sorted(res_hash.keys())
        # res_var_list.sort()
        for var in res_var_list:
            #print "var", var, res_hash[var]
            if not var.startswith("xs"):
                continue
            m = pat1.match(var)
            if not m:
                sys.stderr.write("cannot parse var: %s\n" % (var))
                sys.exit(1)
            else:
                # if options.verbose:
                #     print ("sol:", var, res_hash[var] )
                jobid = int(m.group(1))
                pid   = int(m.group(2))
                val  = float(res_hash[var])
                if jobid not in x: # x.has_key(jobid):
                    x[jobid] = {}
                x[jobid][pid] = val


        for jobid in x.keys():

            node = self.fakeId2compNode[jobid]
            vtasks = vtask_hash[node.id]

            for pid in x[jobid].keys():

                if pid < vtasks.get_nb_vtasks():
                    # pid  1..Mj
                    xstar = x[jobid][pid]
                    # but here we need to subtract -1 as we start counting from 0
                    vxj   = vtasks.get_execution_time_at_idx(pid-1)
                    vxj1  = vtasks.get_execution_time_at_idx(pid)

                    # if options.verbose:
                    #     print "round", jobid, ",", pid, ",", x[jobid][pid]
                    # if options.verbose:
                    #     print ("jobid:", jobid, " pid:", pid, " xstar:", xstar, " vxj1:", vxj1, " vxj:", vxj)
                    if (xstar - vxj1) < RHO * (vxj - vxj1):
                        x[jobid][pid] = 0.0
                    else:
                        x[jobid][pid] = vxj

                    #if options.verbose:
                    #     print "to", jobid, ",", pid, ",", x[jobid][pid]

                #else:
                #    if options.verbose:
                #        print ">>> to", jobid, ",", pid, ",", x[jobid][pid]


            max_time = -1.0
            max_p    = 0
            for pid in x[jobid].keys():
                if (x[jobid][pid] > max_time) or (x[jobid][pid] == max_time and pid < max_p):
                    max_p = pid
                    max_time = x[jobid][pid]

            # if options.verbose:
            #     print ("rounded", jobid, "(real=", node.id ,")" , " to nb_procs: ", max_p, " time:", max_time)

            res_p = min( MU, max_p )
            res_time = get_time_of_node(node, res_p, grid)

            solution[(node.id)] = [ res_p, res_time ]


        return solution



