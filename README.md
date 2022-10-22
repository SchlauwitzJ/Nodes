# Nodes
This repo contains the classes for Nodes and Edges that can by used to contruct graphs and trees. Each node and edge can be assigned properties that can also be used as search conditions. Targets are in the form of {'node property key': {'condition operator': value}}, subject to conditions have the same format but use the edge properties. If no target conditions are given, it will return all nodes that can be reached from the starting point subject to the edge traversal conditions.

To improve search speed, asyncio is used to evaluate multiple branches at once.

# Todo
Enable saving to and loading from file.
