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

'''
Created on Nov 28, 2010

@author: sascha
'''
import networkx as nx


class MixteSimDAG:

    def __init__(self):
        self.node_hash = {}

    def add_node(self, node):
        assert (node != None)
        self.node_hash[node.id] = node

    def get_nodes(self):
        return self.node_hash.values()

    def get_computational_nodes(self):
        return self.node_hash.values()

    # def get_computational_nodes(self):
    #     nodes = self.node_hash.values()
    #     nodes = filter(lambda x: x.type == "COMPUTATION", nodes)
    #     return nodes

    def get_computational_children(self, nid):
        children = []
        for key in self.node_hash[nid].get_children():
            children.append(self.node_hash[key])
        return children

    # def get_computational_children(self, nid):
    #     child_hash = {}  # id -> child
    #     for childid in self.node_hash[nid].get_children():
    #         child = self.node_hash[childid]
    #         if child.get_type() == "COMPUTATION":
    #             child_hash [ child.id ] = child
    #         elif child.get_type() == "TRANSFER":
    #             for grand_childid in self.node_hash[child.id].get_children():
    #                 grandchild = self.node_hash[grand_childid]
    #                 if grandchild.get_type() == "COMPUTATION":
    #                     child_hash[ grandchild.id ] = grandchild
    #     return child_hash.values()

    def get_node_by_id(self, node_id):
        assert (id != None)
        return self.node_hash[node_id]

    def get_root(self):
        root_node = None
        for node in self.node_hash.values():
            if node.type == "ROOT":
                root_node = node
                break
        return root_node

    def __repr__(self):
        str = "#nodes %d\n" % (len(self.node_hash.keys()))
        nodeids = self.node_hash.keys()
        nodeids.sort()
        for nodeid in nodeids:
            node = self.node_hash[nodeid]
            str += repr(node)
        return str


class Node:

    def __init__(self, id, type, value, alpha):
        self.id = id
        self.type = type
        self.value = value
        self.alpha = alpha
        self.children = {}

    def add_child(self, child_id):
        self.children[child_id] = None

    def get_children(self):
        return self.children.keys()

    def get_id(self):
        return self.id

    def get_type(self):
        return self.type

    def get_alpha(self):
        return self.alpha

    def get_value(self):
        return self.value

    def __repr__(self):
        str = "node id=%s value=%g alpha=%g\n" % (self.id, self.value, self.alpha)
        str += "children :"
        for child in self.children.keys():
            str += " %s" % (child)
        str += "\n"
        return str

    # id    = property(get_id)
    # type  = property(get_type)
    # alpha = property(get_alpha)
    # value = property(get_value)
    #
    # @id.setter
    # def id(self, value):
    #     self._id = value
    # @type.setter
    # def type(self, value):
    #     self._type = value
    #
    # @alpha.setter
    # def alpha(self, value):
    #     self._alpha = value
    #
    # @value.setter
    # def value(self, value):
    #     self._value = value


class DagParser:

    def __init__(self):
        pass

    def parse(self, tG):
        dag = MixteSimDAG()
        for nd in tG.nodes:
            node = Node(nd, "COMPUTATION", tG.nodes[nd]['weight'], tG.nodes[nd]['alpha'])
            for succ in tG.successors(nd):
                node.add_child(succ)
            dag.add_node(node)
        return dag

    # def parse(self, filename):
    #
    #     assert( filename != None )
    #
    #     fh = open(filename)
    #     content = fh.readlines()
    #     fh.close()
    #
    #     dag = MixteSimDAG()
    #
    #     for line in content:
    #
    #         if len(line) > 0 and line[0] == "#":
    #             continue
    #
    #         token = line.split()
    #         #print token
    #
    #         if token[0] == "NODE_COUNT":
    #             continue
    #
    #         assert( token[0] == "NODE" )
    #
    #         node_id = token[1]
    #
    #         children_list = token[2]
    #         children = children_list.split(",")
    #
    #         type  = token[3]
    #
    #         if token[4] != "NO":
    #             value = float(token[4])
    #         else:
    #             value = token[4]
    #
    #         alpha = float(token[5])
    #
    #         node = Node(node_id, type, value, alpha)
    #         for child_id in children:
    #             if child_id != "-":
    #                 node.add_child(child_id)
    #
    #         dag.add_node(node)
    #
    #     return dag


class MixteSimPlatform:

    def __init__(self, p, speed):
        self.p = p
        self.speed = speed

    def get_p(self):
        return self.p

    def get_speed(self):
        return self.speed

    # p = property(get_p)
    # speed = property(get_speed)


class MixteSimPlatformParser:

    def __init__(self):
        pass

    def parse(self, P):
        # assert( filename != None )
        #
        # fh = open(filename)
        # content = fh.readlines()
        # fh.close()
        #
        # p = None
        # speed = None
        #
        # for line in content:
        #
        #     if len(line) > 0 and line[0] == "#":
        #         continue
        #
        #     tokens = line.split()
        #
        #     if len(tokens) < 1:
        #         continue
        #
        #     if tokens[0] == "HOST_COUNT":
        #         p = int( tokens[1] )
        #     elif tokens[0] == "HOST":
        #         speed = float( tokens[7] )
        #         break
        #
        # assert( p != None )
        # assert( speed != None )

        platform = MixteSimPlatform(P, 1)
        return platform
