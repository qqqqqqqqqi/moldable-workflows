
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

import sys

# p is here current size of allocation (NOT total num of processors)

def get_time_of_node(node, p, grid):
    # speed = grid.get_speed()
    # if p > grid.get_p():
    #     sys.stderr.write("p (%d) > larger than grid size: %d\n" % (p, grid.get_p()))
    #     sys.exit(1)
    # if node.get_alpha() >= 1:
    #     sys.stderr.write("warning: node %s has alpha of %g\n" % (node.id, node.get_alpha()))
    #     sys.exit(1)
    time = 0
    if isinstance(node.get_value(),list):
        sizes=node.get_value()
        alphas=node.get_alpha()
        for i in range(len(sizes)):
            seq_ops=sizes[i]*alphas[i]
            par_ops=sizes[i]*(1-alphas[i])/float(p)
            time+=(seq_ops+par_ops)
    else :
        seq_ops = node.get_value() * node.get_alpha()
        par_ops = node.get_value() * (1 - node.get_alpha()) / float(p)
        time = (seq_ops + par_ops)
    #print "time of %s with p=%d => %g" % (node.id, p, time)
    return time
