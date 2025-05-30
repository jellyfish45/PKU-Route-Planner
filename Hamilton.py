import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from gurobipy import Model, GRB
import gurobipy as gp

class HamiltonProgramming:
    def __init__(self):
        self.G = nx.MultiGraph()  # 创建一个多重图
        self.ind2vertex = {}  # 用于将索引映射到顶点
        self.vertex2ind = {}  # 用于将顶点映射到索引
        self.adjacency_matrix = None

    # 添加边
    def addEdge(self, u, v):
        self.G.add_edge(u, v)
        self.update_matrices()

    # 添加多个边
    def addEdges(self, edges):
        self.G.add_edges_from(edges)
        self.update_matrices()

    # 更新邻接矩阵和顶点索引列表/字典
    def update_matrices(self):
        for ind, vertex in enumerate(self.G.nodes()):
            self.ind2vertex[ind] = vertex
            self.vertex2ind[vertex] = ind
        self.ind = list(self.ind2vertex.keys())  # 获取顶点的索引列表
        self.adjacency_matrix = nx.adjacency_matrix(self.G).todense()

    def HamiltonPath_solver(self):
        from gurobipy import Model, GRB, quicksum
    
        num_nodes = len(self.ind)
        nodes = self.ind
        arcs = [(i, j) for i in nodes for j in nodes if i != j and self.adjacency_matrix[i, j] > 0]
    
        m = Model("HamiltonPath")
        m.setParam('OutputFlag', 0)
    
        # 二进制变量 x[i, j]：是否从 i 走到 j
        x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    
        # 顺序变量：MTZ formulation to eliminate subtours
        u = m.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0, ub=num_nodes - 1, name="u")
    
        # 每个点最多有一个入度和一个出度
        for j in nodes:
            m.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in arcs) <= 1, f"in_deg_{j}")
            m.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in arcs) <= 1, f"out_deg_{j}")
    
        # 总共 n-1 条边
        m.addConstr(quicksum(x[i, j] for i, j in arcs) == num_nodes - 1, "total_edges")
    
        # MTZ 去除子环（子路径）
        for i in nodes:
            for j in nodes:
                if i != j and (i, j) in arcs:
                    m.addConstr(u[i] - u[j] + num_nodes * x[i, j] <= num_nodes - 1, f"mtz_{i}_{j}")
    
        m.setObjective(1, GRB.MAXIMIZE)
        m.optimize()
    
        if m.status == GRB.OPTIMAL:
            print("数学规划判断：存在哈密顿通路")
            edges = [(i, j) for i, j in arcs if x[i, j].X > 0.5]
    
            # 将边序列整理为路径
            path = []
            successors = {i: j for i, j in edges}
            start = [i for i in nodes if all((j, i) not in edges for j in nodes)][0]
            print(start)
            while start in successors:
                path.append((start, successors[start]))
                start = successors[start]

            self.begin = path[0][0]
            self.end = path[-1][1]
    
            self.get_hamiltonian_path(x)
            path = self.hamiltonian_path 
            print(" -> ".join([str(u) for u, v in path] + [str(path[-1][1])]))
        else:
            print("数学规划判断：不存在哈密顿通路")

    def get_hamiltonian_path(self, x):
        self.hamiltonian_path = []
        visited_nodes = {}  # 使用字典记录每个节点的访问次数
        visited_edges = {}  # 使用字典记录每条边的访问次数

        # 初始化访问计数
        for (u, v) in x.keys():
            visited_edges[(u, v)] = 0
            visited_edges[(v, u)] = 0  # 记录反向边
        for u in self.ind:
            visited_nodes[u] = 0

        class PathFound(Exception):
            # 利用异常处理以退出整个dfs搜索
            pass

        def dfs(current_vertex):
            # 所有边都已访问，且到达self.end
            if all(visited_edges[(u, v)] == x[u, v].X for (u, v) in x.keys()) and current_vertex == self.end:
                # 所有边都已访问，且到达终点
                raise PathFound()

            visited_nodes[current_vertex] += 1
            # 查找邻居
            for neighbor in [self.vertex2ind[i] for i in self.G.neighbors(self.ind2vertex[current_vertex])]:
                if x[current_vertex, neighbor].X > visited_edges[(current_vertex, neighbor)]:
                    self.hamiltonian_path.append((current_vertex, neighbor))
                    visited_edges[(current_vertex, neighbor)] += 1
                    dfs(neighbor)
                    # 回溯
                    visited_edges[(current_vertex, neighbor)] -= 1
                    self.hamiltonian_path.pop()

            visited_nodes[current_vertex] -= 1

        try:
            dfs(self.begin)
        except PathFound:
            # 将索引映射回顶点
            for i in range(len(self.hamiltonian_path)):
                u, v = self.hamiltonian_path[i]
                self.hamiltonian_path[i] = (
                    self.ind2vertex[u], self.ind2vertex[v])
            pass  # 不处理异常，直接退出
