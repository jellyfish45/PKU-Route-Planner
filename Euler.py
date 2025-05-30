import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from gurobipy import Model, GRB
import gurobipy as gp
class EulerProgramming:
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

    def EulerPath_solver(self):
        num_nodes = len(self.ind)
        s = self.ind[0]  # 选择第一个节点作为源节点

        # 创建边的列表
        arcs = []
        for i in self.ind:
            for j in self.ind:
                if i != j and self.adjacency_matrix[i, j] > 0:
                    arcs.append((i, j))

        # 首先判断图是否连通
        m = Model("ConnectedGraph")
        m.setParam('OutputFlag', 0)  # 关闭求解器输出
        f = m.addVars(arcs, lb=0, vtype=GRB.CONTINUOUS, name="f")

        # 容量约束
        for i, j in arcs:
            m.addConstr(f[i, j] + f[j, i] <= (num_nodes - 1) * self.adjacency_matrix[i, j],
                        name=f"flow_constraint_{i}_{j}")

        # 流量平衡约束
        # 对于源节点 s
        m.addConstr(
            gp.quicksum(f[s, j] for j in self.ind if (s, j) in arcs) -
            gp.quicksum(f[j, s] for j in self.ind if (
                j, s) in arcs) == (num_nodes - 1),
            name="flow_conservation_s"
        )

        # 对于其他节点
        for i in self.ind:
            if i != s:
                m.addConstr(
                    gp.quicksum(f[i, j] for j in self.ind if (i, j) in arcs) -
                    gp.quicksum(f[j, i]
                                for j in self.ind if (j, i) in arcs) == -1,
                    name=f"flow_conservation_{i}"
                )

        m.setObjective(1, GRB.MAXIMIZE)
        m.optimize()
        # 如果环路不连通，则直接判断不存在欧拉路径
        if m.status != GRB.OPTIMAL:
            print("数学规划判断：不存在欧拉路径")
            return

        # 如果环路连通，则继续判断是否存在欧拉路径
        m = Model("EulerPath")
        m.setParam('OutputFlag', 0)  # 关闭求解器输出
        x = m.addVars(arcs, vtype=GRB.INTEGER, name="x")
        # 创建二进制变量 y 和 z
        y = m.addVars(self.ind, vtype=GRB.BINARY, name="y")
        z = m.addVars(self.ind, vtype=GRB.BINARY, name="z")

        # 添加 y 和 z 变量的约束
        m.addConstr(gp.quicksum(y[i] for i in self.ind) == 1, "sum_y_equals_1")
        m.addConstr(gp.quicksum(z[i] for i in self.ind) == 1, "sum_z_equals_1")

        # 添加流量平衡约束
        for i in self.ind:
            out_flow = gp.quicksum(x[i, j]
                                    for j in self.ind if (i, j) in arcs)
            in_flow = gp.quicksum(x[j, i]
                                    for j in self.ind if (j, i) in arcs)
            m.addConstr(out_flow - in_flow ==
                        y[i] - z[i], name=f"flow_balance_{i}")

        # 边容量约束
        for i, j in arcs:
            m.addConstr(x[i, j] <= self.adjacency_matrix[i, j],
                        name=f"capacity_{i}_{j}")
            m.addConstr(x[i, j] + x[j, i] == self.adjacency_matrix[i, j],
                        name=f"edge_flow_{i}_{j}")

        m.setObjective(1, GRB.MAXIMIZE)
        m.optimize()

        # 分析结果
        if m.status == GRB.OPTIMAL:
            print("数学规划判断：存在欧拉路径")
            for i in self.ind:
                if y[i].X > 0:
                    self.begin = i
                if z[i].X > 0:
                    self.end = i
            self.get_eulerian_path(x)
            print(" -> ".join([str(i) for i in self.eulerian_path]))
        else:
            print("数学规划判断：不存在欧拉路径")
            
    def get_eulerian_circuit(self, x):
        self.eulerian_circuit = []
        visited_edges = {}  # 使用字典记录每条边的访问次数
        visited_nodes = {}  # 使用字典记录每个节点的访问次数

        # 初始化访问计数
        for (u, v) in x.keys():
            visited_edges[(u, v)] = 0
            visited_edges[(v, u)] = 0  # 如果是无向图，也需要记录反向边
        for u in self.ind:
            visited_nodes[u] = 0

        class CircuitFound(Exception):
            # 利用异常处理以退出整个dfs搜索
            pass

        def dfs(current_vertex):
            visited_nodes[current_vertex] += 1
            # 在邻居结点中查找
            for neighbor in [self.vertex2ind[i] for i in list(self.G.neighbors(self.ind2vertex[current_vertex]))]:
                if x[current_vertex, neighbor].X > visited_edges[(current_vertex, neighbor)]:
                    self.eulerian_circuit.append((current_vertex, neighbor))
                    visited_edges[(current_vertex, neighbor)] += 1

                    if all(value > 0 for value in visited_nodes.values()) and \
                            all(visited_edges[(u, v)] == x[u, v].X for (u, v) in x.keys()) and \
                            all(visited_edges[(v, u)] == x[v, u].X for (u, v) in x.keys()):
                        # 所有结点和所有边都得到访问时，则退出dfs
                        raise CircuitFound()

                    dfs(neighbor)
                    visited_edges[(current_vertex, neighbor)] -= 1
                    self.eulerian_circuit.pop()

            visited_nodes[current_vertex] -= 1

        try:
            current_vertex = self.ind[0]
            dfs(current_vertex)
        except CircuitFound:
            for i in range(len(self.eulerian_circuit)):
                self.eulerian_circuit[i] = (
                    self.ind2vertex[self.eulerian_circuit[i][0]], self.ind2vertex[self.eulerian_circuit[i][1]])  # 将索引映射回顶点
            pass  # 不处理异常，直接退出

    def get_eulerian_path(self, x):
        self.eulerian_path = []
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
                    self.eulerian_path.append((current_vertex, neighbor))
                    visited_edges[(current_vertex, neighbor)] += 1
                    dfs(neighbor)
                    # 回溯
                    visited_edges[(current_vertex, neighbor)] -= 1
                    self.eulerian_path.pop()

            visited_nodes[current_vertex] -= 1

        try:
            dfs(self.begin)
        except PathFound:
            # 将索引映射回顶点
            for i in range(len(self.eulerian_path)):
                u, v = self.eulerian_path[i]
                self.eulerian_path[i] = (
                    self.ind2vertex[u], self.ind2vertex[v])
            pass  # 不处理异常，直接退出