import io
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

class LinProgSolver:
    
    def solve(self, lp_fname, sol_fname):
        pass
    
    def read_result(self, sol_fname):
        pass
    
class LpSolutionProducer:
    
    def get_solution(self, res_hash, dag, grid, options):
        pass    
    
    
class WriteBuffer:
    
    def __init__(self):
        self.output = io.StringIO()
        self.cnum = 0
    
    def write(self, s):
        self.output.write(s)
        
    def writec(self, s):
        self.write_constraint(s)
        
    def write_constraint(self, s):
        const = "subject to c%d: %s" % ( self.cnum, s )
        self.output.write(const)
        self.cnum += 1

    def write_constraint_with_option(self, s, option):
        const = "subject to c%d %s: %s" % ( self.cnum, option, s )
        self.output.write(const) 
        self.cnum += 1
    
    def __repr__(self):
        return self.output.getvalue()

    def __str__(self):
        return self.__repr__()
    
    def __del__(self):
        self.output.close()

class WriteBufferCplex(WriteBuffer):

    def write_constraint(self, s):
        self.output.write(s)
    
    
# store variable to declare later    
class VarStore:    
    
    def __init__(self):
        self.store = []
        
    def add(self, var_name):
        self.store.append(var_name)
        
    def get_variables(self):
        return self.store
    