import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from gurobipy import Model, GRB, quicksum
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
        """
        edges: List[Tuple[str, str, float]]，使用字符串名称
        """
        for u, v, w in edges:
            self.G.add_edge(u, v, weight=w)
        self.update_matrices()
        
        # 将 edge_list 转换为使用索引的格式，供 solver 用
        self.edge_list = [
            (self.vertex2ind[u], self.vertex2ind[v], w)
            for u, v, w in edges
        ]
        print("[调试] 原始边列表:", edges)
        print("[调试] 索引化后的边列表:", self.edge_list)
        print("[调试] vertex2ind:", self.vertex2ind)



    # 更新邻接矩阵和顶点索引列表/字典
    def update_matrices(self):
        for ind, vertex in enumerate(self.G.nodes()):
            self.ind2vertex[ind] = vertex
            self.vertex2ind[vertex] = ind
        self.ind = list(self.ind2vertex.keys())  # 获取顶点的索引列表
        self.adjacency_matrix = nx.adjacency_matrix(self.G).todense()
        print("vertex2ind:", self.vertex2ind)
        print("ind2vertex:", self.ind2vertex)


    def HamiltonPath_solver(self):
        nodes = self.ind
        num_nodes = len(nodes)
    
        # 假设 self.edge_list 是 List[Tuple[int, int, float]]，允许多重边
        # 构建所有弧（每条无向边拆成两个有向边）
        arcs = []        # List of (i, j, edge_id)
        weights = {}     # Dict with key (i, j, edge_id) -> weight
    
        for edge_id, (i, j, w) in enumerate(self.edge_list):
            if i == j:
                continue  # 不考虑自环
            arcs.append((i, j, edge_id))
            arcs.append((j, i, edge_id))
            weights[(i, j, edge_id)] = w
            weights[(j, i, edge_id)] = w
    
        m = Model("HamiltonPath_Multigraph")
        m.setParam('OutputFlag', 0)
    
        # 决策变量：每条有向边是否被选中
        x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
    
        # MTZ变量消除子环
        u = m.addVars(nodes, vtype=GRB.CONTINUOUS, lb=0, ub=num_nodes - 1, name="u")
    
        # 每个节点的度为 2（每条无向边拆成两个方向）
        for j in nodes:
            in_deg = quicksum(x[i, j, eid] for (i, j2, eid) in arcs if j2 == j)
            out_deg = quicksum(x[j, k, eid] for (j2, k, eid) in arcs if j2 == j)
            m.addConstr(in_deg + out_deg == 2, f"degree_{j}")
    
        # MTZ 子环消除约束
        for (i, j, eid) in arcs:
            if i != j:
                m.addConstr(u[i] - u[j] + num_nodes * x[i, j, eid] <= num_nodes - 1, f"mtz_{i}_{j}_{eid}")
    
        # 最小化总权重
        m.setObjective(quicksum(weights[i, j, eid] * x[i, j, eid] for (i, j, eid) in arcs), GRB.MINIMIZE)
    
        m.optimize()
    
        if m.status == GRB.OPTIMAL:
            print("存在哈密顿路径，已找到最短路径")
            selected = [(i, j) for (i, j, eid) in arcs if x[i, j, eid].X > 0.5]
    
            # 构造邻接表
            from collections import defaultdict
            successors = defaultdict(list)
            in_degree = {node: 0 for node in nodes}
            for i, j in selected:
                successors[i].append(j)
                in_degree[j] += 1
    
            # 找起点
            start_candidates = [n for n in nodes if in_degree[n] == 1]
            if len(start_candidates) == 2:
                start = start_candidates[0]
            elif len(start_candidates) == 0:
                start = nodes[0]  # 是一个哈密顿环
            else:
                print("错误：路径端点数异常")
                return
    
            # 还原路径
            path = [start]
            visited = set(path)
            current = start
            while True:
                next_nodes = [nbr for nbr in successors[current] if nbr not in visited]
                if not next_nodes:
                    break
                current = next_nodes[0]
                path.append(current)
                visited.add(current)
    
            ppath_names = []
            for n in path:
                if isinstance(n, int):
                    name = self.ind2vertex.get(n, f"[未知:{n}]")
                else:
                    name = n
                ppath_names.append(name)
            print("路径：" + " -> ".join(ppath_names))
            self.hamiltonian_path = [(ppath_names[i], ppath_names[i + 1]) for i in range(len(ppath_names) - 1)]
    
        else:
            print("不存在哈密顿路径")