from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import asyncio
import numpy as np
import time
from collections import Counter
from SpatialSystems.Geometric import Geo
import json

# FLAGS = {}
# {'search_key': }
NODE_RECORDS = {}


def edge_name(node1: Node, node2: Node):
    return f"{node1.node_id()}<--->{node2.node_id()}"


@dataclass
class Edge:
    def __init__(self, node1: Node, node2: Node):
        """
        :param node1:
        :param node2:
        """
        self.node1 = node1
        self.node2 = node2
        self.one_to_two = {"value": 0, "weight": 1, "path length": 1}
        self.two_to_one = {"value": 0, "weight": 1, "path length": 1}

        # assign the connection locally in node1 but here for node2
        # self.node1.connections[edge_name(node1, node2)] = self
        self.node2.connections[edge_name(node2, node1)] = self

    def switch(self, node1: Node = None, node2: Node = None):
        if node1 is not None:
            self.node1 = node1
        if node2 is not None:
            self.node2 = node1
        elif node1 is None:  # implies node2 is None
            node_tmp = self.node1
            self.node1 = self.node2
            self.node2 = node_tmp
        return

    def opposite(self, node: Node):
        if node.node_id() not in (self.node1.node_id(), self.node2.node_id()):
            return self.node1.node_id(), self.node2.node_id()
        elif node.node_id() == self.node1.node_id() and node.node_id() == self.node2.node_id():
            return None
        elif node.node_id() == self.node1.node_id():
            return self.node2
        else:
            return self.node1

    async def traverse_search(self, existing_path: dict, subject_to: dict[str, dict] = None,
                              subject_to_delta: dict[str, dict] = None,
                              targets=None, glob_ky=None):
        """
        :param subject_to_delta:
        :param glob_ky:
        :param targets:
        :param existing_path: contains the cumulative data for the metric
        :param subject_to: {metric: {'<': value}}
        :return:
        """
        if subject_to is None:
            subject_to = {}
        if subject_to_delta is None:
            subject_to_delta = {}

        new_path_data = deepcopy(existing_path)

        if existing_path['nodes'][-1] == self.node1.node_id():
            pdata = self.one_to_two
            next_node = self.node2
        else:
            pdata = self.two_to_one
            next_node = self.node1

        traversable = True
        for ky in pdata.keys():
            if ky in new_path_data['sums'].keys():
                # integrate edge values with path total
                new_path_data['sums'][ky] += pdata[ky]
            else:
                new_path_data['sums'][ky] = pdata[ky]
            new_path_data['delta_values'][ky] = pdata[ky]

            if traversable:
                if ky in subject_to.keys():
                    # it is something we are interested in and this path has yet to be ruled out
                    for con_op, val in subject_to[ky].items():
                        if con_op == '<':
                            traversable = new_path_data['sums'][ky] < val
                        elif con_op == '<=':
                            traversable = new_path_data['sums'][ky] <= val
                        elif con_op == '>':
                            traversable = new_path_data['sums'][ky] > val
                        elif con_op == '>=':
                            traversable = new_path_data['sums'][ky] >= val
                        elif con_op == '==':
                            traversable = new_path_data['sums'][ky] == val
                        elif con_op == '!=':
                            traversable = new_path_data['sums'][ky] != val
                        if not traversable:
                            break

                if ky in subject_to_delta.keys():
                    # it is something we are interested in and this path has yet to be ruled out
                    for con_op, val in subject_to_delta[ky].items():
                        if con_op == '<':
                            traversable = new_path_data['delta_values'][ky] < val
                        elif con_op == '<=':
                            traversable = new_path_data['delta_values'][ky] <= val
                        elif con_op == '>':
                            traversable = new_path_data['delta_values'][ky] > val
                        elif con_op == '>=':
                            traversable = new_path_data['delta_values'][ky] >= val
                        elif con_op == '==':
                            traversable = new_path_data['delta_values'][ky] == val
                        elif con_op == '!=':
                            traversable = new_path_data['delta_values'][ky] != val
                        if not traversable:
                            break

        if traversable:
            # evaluate the next node
            return await next_node.search_for(targets=targets,
                                              current_path=new_path_data,
                                              subject_to=subject_to, use_glob_ky=glob_ky)
        elif targets is None:
            # give connected region within the traversable limits instead
            return new_path_data

        return None

    def __del__(self):
        edge1 = edge_name(self.node1, self.node2)
        edge2 = edge_name(self.node2, self.node1)
        if edge1 in self.node1.connections.keys():
            del self.node1.connections[edge1]
        if edge2 in self.node2.connections.keys():
            del self.node2.connections[edge2]


@dataclass
class Node:
    def __init__(self, node_id):
        self.properties = {"node_id": node_id}
        self.connections = {}

    def node_id(self):
        return self.properties['node_id']

    def connect_to(self, node: Node):
        edge = edge_name(self, node)
        self.connections[edge] = Edge(node1=self, node2=node)
        # print(edge_name(self, node))
        return

    def disconnect_from(self, node):
        edge = edge_name(self, node)
        if edge in self.connections.keys():
            del self.connections[edge]
        return

    async def search_for(self, targets: dict[str, dict] = None, current_path: dict = None,
                         subject_to: dict[str, dict] = None, subject_to_delta: dict[str, dict] = None,
                         use_glob_ky=None):
        """

        :param subject_to_delta: edge value conditions
        :param use_glob_ky: the key to be used for the search
        :param targets: node point based targets (satisfying 1 will result in is_found = True
        :param current_path:
        :param subject_to: cumulative edge conditions
        :return:
        """

        if use_glob_ky is None:
            glob_ky = f'{self.node_id()}.search_for({json.dumps(targets)})'
        else:
            glob_ky = use_glob_ky

        if current_path is None:
            # the starting point of our search
            this_path = {'nodes': [self.node_id()], 'sums': {}, 'values': {}, 'delta_values': {}}
            NODE_RECORDS[glob_ky] = {'nodes': [self.node_id()], 'is_found': False}
        else:
            # continuing the search with existing data
            this_path = deepcopy(current_path)
            this_path['nodes'].append(self.node_id())
            NODE_RECORDS[glob_ky]['nodes'].append(self.node_id())

        # ===========================================
        if subject_to is None:
            loc_conditions = {}
        else:
            loc_conditions = subject_to

        # ===========================================
        is_found = False
        if targets is not None:
            # check if any target conditions have been reached
            # each check acts as an AND therefore all conditions must be true
            cntr = 0
            t_cntr = 0
            for prprty, remndr in targets.items():
                node_prop = self.properties.get(prprty)
                this_path['values'][prprty] = {}
                for oper, val in remndr.items():
                    cntr += 1
                    if ((oper == '<' and node_prop < val) or
                            (oper == '<=' and node_prop <= val) or
                            (oper == '>' and node_prop > val) or
                            (oper == '>=' and node_prop >= val) or
                            (oper == '==' and node_prop == val) or
                            (oper == '!=' and node_prop != val)):
                        t_cntr += 1
                        this_path['values'][prprty][oper] = node_prop
            if cntr == t_cntr:
                # a valid target node matches all target conditions
                is_found = True
                NODE_RECORDS[glob_ky]['is_found'] = True
        # else we are looking for the earliest fail condition from an edge

        # ===========================================
        if (NODE_RECORDS[glob_ky]['is_found'] or len(self.connections) == 1 < len(this_path['nodes'])
                or len(self.connections) == 0):
            # dead end or island node
            # this is here so that it gets processed asap.
            # solution found. Return without further processing
            pass
        elif len(self.connections) <= 1:
            # do not waste time making coroutines
            for edge in self.connections.values():

                next_node = edge.opposite(node=self)
                if isinstance(next_node, Node) and next_node.node_id() not in NODE_RECORDS[glob_ky]['nodes']:
                    # next node is valid and unique to this search
                    searched_path = await edge.traverse_search(existing_path=this_path, subject_to=loc_conditions,
                                                               subject_to_delta=subject_to_delta, targets=targets,
                                                               glob_ky=glob_ky)

                    # print(searched_path)
                    if searched_path is not None:
                        # we found the path here, update the returned path results
                        this_path = searched_path
                        is_found = True
                        break
                if NODE_RECORDS[glob_ky]['is_found']:
                    # the path has been found elsewhere, do not continue
                    break

        else:
            # continue down the graph
            tasks = []
            for edge in self.connections.values():
                next_node = edge.opposite(node=self)

                if isinstance(next_node, Node) and next_node.node_id() not in NODE_RECORDS[glob_ky]['nodes']:
                    # prevent overlapping paths
                    tasks.append(asyncio.create_task(
                        edge.traverse_search(existing_path=this_path, subject_to=loc_conditions,
                                             subject_to_delta=subject_to_delta, targets=targets, glob_ky=glob_ky)))

            if not NODE_RECORDS[glob_ky]['is_found']:
                results = await asyncio.gather(*tasks)
                for searched_path in results:
                    # print(searched_path)
                    if searched_path is not None:
                        # return the first path found chronologically
                        this_path = searched_path
                        is_found = True
                        break

        if current_path is None:
            if targets is None:
                # return the regions connected to the start node
                this_path = {'nodes': NODE_RECORDS[glob_ky]['nodes']}
                is_found = True

            if use_glob_ky is None:
                # clear the global search data
                del NODE_RECORDS[glob_ky]

        if is_found:
            return this_path
        return None


async def graph_test():
    n_num = 100
    nodes = [Node(node_id=0)]
    for ind in range(1, n_num):
        nodes.append(Node(node_id=ind))
        nodes[-1].connect_to(node=nodes[np.random.randint(0, ind)])
    start_t = time.time()
    obj_node = np.random.randint(0, n_num)
    my_glb_ky = f'{obj_node}_searching'
    search_path = await nodes[0].search_for(targets={'node_id': {'==': obj_node}},
                                            subject_to={'value': {'==': 0}, 'weight': {'<': 90},
                                                        'path length': {'<=': 90}},
                                            use_glob_ky=my_glb_ky)
    end_t = time.time()
    was_found = False
    search_len = 1
    if search_path is not None:
        was_found = True
        search_len = len(search_path['nodes'])

    print(f"Found:\t{was_found} \nLen:\t{search_len} \nTime:\t{end_t - start_t} "
          f"\nBranching Factor: {len(NODE_RECORDS[my_glb_ky]['nodes'])/search_len} \nObjective: {obj_node}")
    # print(dict(Counter(search_path['nodes'])))
    print(search_path)
    print(NODE_RECORDS[my_glb_ky]['nodes'])
    return

if __name__ == '__main__':
    asyncio.run(graph_test())
