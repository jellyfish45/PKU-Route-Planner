import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
from collections import defaultdict

class HamiltonProgramming:
    def __init__(self):
        self.G = nx.MultiGraph()
        self.ind2vertex = {}
        self.vertex2ind = {}
        self.adjacency_matrix = None
        self.edge_list = []
        self.hamiltonian_path = None

    def addEdge(self, u, v, weight=1.0):
        if u != v:  # 过滤自环
            self.G.add_edge(u, v, weight=weight)
            self.update_matrices()
            if u in self.vertex2ind and v in self.vertex2ind:
                self.edge_list.append((self.vertex2ind[u], self.vertex2ind[v], weight))
            else:
                print(f"警告：顶点 {u} 或 {v} 未在索引中，需先更新矩阵")

    def addEdges(self, edges):
        for u, v, w in edges:
            if u != v:  # 过滤自环
                self.G.add_edge(u, v, weight=w)
        self.update_matrices()
        self.edge_list = [
            (self.vertex2ind[u], self.vertex2ind[v], w)
            for u, v, w in edges if u != v
        ]
        print("[调试] 原始边列表:", edges)
        print("[调试] 索引化后的边列表:", self.edge_list)
        print("[调试] vertex2ind:", self.vertex2ind)

    def update_matrices(self):
        for ind, vertex in enumerate(self.G.nodes()):
            self.ind2vertex[ind] = vertex
            self.vertex2ind[vertex] = ind
        self.ind = list(self.ind2vertex.keys())
        self.adjacency_matrix = nx.adjacency_matrix(self.G).todense()
        self.edge_list = [
            (self.vertex2ind[u], self.vertex2ind[v], self.G[u][v][key].get('weight', 1.0))
            for u, v, key in self.G.edges(keys=True)
        ]
        print("vertex2ind:", self.vertex2ind)
        print("ind2vertex:", self.ind2vertex)
        print("[调试] 索引化后的边列表:", self.edge_list)

    def HamiltonPath_solver(self):
        nodes = self.ind
        num_nodes = len(nodes)

        # 检查图连通性（无向图）
        G_temp = nx.Graph()
        G_temp.add_weighted_edges_from([(self.ind2vertex[i], self.ind2vertex[j], w) for i, j, w in self.edge_list])
        if not nx.is_connected(G_temp):
            print("错误：图不连通，无法找到哈密顿路径")
            return

        arcs = []
        weights = {}
        for edge_id, (i, j, w) in enumerate(self.edge_list):
            if i == j:
                continue
            arcs.append((i, j, edge_id))
            arcs.append((j, i, edge_id))
            weights[(i, j)] = w
            weights[(j, i)] = w

        m = Model("HamiltonPath_Multigraph")
        m.setParam('OutputFlag', 0)

        x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
        u = m.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0, ub=num_nodes - 1, name="u")

        # 度数约束：起点出度1，终点入度1，其他顶点入度=出度=1
        for j in nodes:
            in_deg = quicksum(x[i, j, eid] for (i, j2, eid) in arcs if j2 == j)
            out_deg = quicksum(x[j, k, eid] for (j2, k, eid) in arcs if j2 == j)
            if j == 0:
                m.addConstr(out_deg == 1, f"start_out_{j}")
                m.addConstr(in_deg == 0, f"start_in_{j}")
            elif j == num_nodes - 1:
                m.addConstr(in_deg == 1, f"end_in_{j}")
                m.addConstr(out_deg == 0, f"end_out_{j}")
            else:
                m.addConstr(in_deg == 1, f"in_degree_{j}")
                m.addConstr(out_deg == 1, f"out_degree_{j}")

        # 防止子回路（MTZ约束）
        for (i, j, eid) in arcs:
            if i != j:
                m.addConstr(u[i] - u[j] + num_nodes * x[i, j, eid] <= num_nodes - 1, f"mtz_{i}_{j}_{eid}")

        m.setObjective(quicksum(weights[i, j] * x[i, j, eid] for (i, j, eid) in arcs), GRB.MINIMIZE)
        m.optimize()

        if m.status == GRB.OPTIMAL:
            print("存在哈密顿路径，已找到最短路径")
            selected = [(i, j) for (i, j, eid) in arcs if x[i, j, eid].X > 0.5]
            
            successors = defaultdict(list)
            in_degree = {node: 0 for node in nodes}
            for i, j in selected:
                successors[i].append(j)
                in_degree[j] += 1

            # 选择起点
            start_candidates = [n for n in nodes if in_degree[n] == 0]
            if not start_candidates:
                print("错误：无有效起点")
                return
            start = start_candidates[0]

            # 构建路径
            path = [start]
            visited = set(path)
            current = start
            try:
                while len(path) < num_nodes:
                    next_nodes = [nbr for nbr in successors[current] if nbr not in visited]
                    if not next_nodes:
                        print("错误：路径中断，无法覆盖所有顶点")
                        return
                    current = next_nodes[0]
                    path.append(current)
                    visited.add(current)
            except IndexError:
                print("错误：索引越界，路径构建失败")
                return

            # 验证路径覆盖
            if len(path) != num_nodes:
                print("错误：路径未覆盖所有顶点")
                return

            path_names = [self.ind2vertex.get(n, f"[未知:{n}]") for n in path]
            print("路径：" + " -> ".join(path_names))
            self.hamiltonian_path = [(path_names[i], path_names[i + 1]) for i in range(len(path_names) - 1)]
        else:
            print("不存在哈密顿路径")
