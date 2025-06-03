import pandas as pd
import streamlit as st
import networkx as nx
from Hamilton import HamiltonProgramming
from Euler import EulerProgramming

def plan_paths(selected_nodes, df, simple_df):
    # 路径过滤
    filtered_df = df[df["起点"].isin(selected_nodes) & df["终点"].isin(selected_nodes)]
    filtered_df = filtered_df[['起点', '终点', '预计步行时间_分钟']]
    filtered_df["起点"] = filtered_df["起点"].str.strip()
    filtered_df["终点"] = filtered_df["终点"].str.strip()


    simple_filtered_df = simple_df[simple_df["起点"].isin(selected_nodes) & simple_df["终点"].isin(selected_nodes)]
    simple_filtered_df = simple_filtered_df[['起点', '终点', '预计步行时间_分钟']]

    # 检查数据完整性
    missing_nodes = [node for node in selected_nodes if node not in filtered_df["起点"].values and node not in filtered_df["终点"].values]
    if missing_nodes:
        st.error(f"❌ 数据中缺少以下景点的路径信息：{missing_nodes}")
        return None, 0, f"数据缺失: {missing_nodes}", None, 0, "无解", filtered_df, simple_filtered_df

    edges = list(filtered_df.itertuples(index=False, name=None))  
    edge_pairs = [(u, v, w) for u, v, w in edges if u != v] + [(v, u, w) for u, v, w in edges if u != v]
    print(f"Edge_pairs:{edge_pairs}")

    if len(edge_pairs) < len(selected_nodes) - 1:
        st.warning("子图边数不足，可能无法形成连通图！")
        return None, 0, "边数不足", None, 0, "无解", filtered_df, simple_filtered_df

    # 检查有向图连通性
    G_temp = nx.DiGraph()
    G_temp.add_weighted_edges_from(edge_pairs)
    G_undirected = G_temp.to_undirected()
    if not nx.is_connected(G_undirected):
        st.error("❌ 子图不连通，无法覆盖所有景点！")
        return None, 0, "子图不连通", None, 0, "无解", filtered_df, simple_filtered_df

    hmp = HamiltonProgramming()
    hmp.addEdges(edge_pairs)

    hamilton_path = None
    hamilton_time = float("inf")
    hamilton_status = "无解"

    try:
        hmp.HamiltonPath_solver()
        if hasattr(hmp, "hamiltonian_path") and hmp.hamiltonian_path:
            path = hmp.hamiltonian_path
            hamilton_time = 0
            missing_edges = []
            for u, v in path:
                sub = filtered_df[((filtered_df["起点"] == u) & (filtered_df["终点"] == v)) |
                                  ((filtered_df["起点"] == v) & (filtered_df["终点"] == u))]
                if not sub.empty:
                    hamilton_time += sub["预计步行时间_分钟"].min()
                else:
                    print(f"[❌ Missing edge] ({u}, {v}) not found in filtered_df")
                    missing_edges.append((u, v))
            if not missing_edges and len(set(u for u, _ in path).union({path[-1][1]})) == len(selected_nodes):
                hamilton_path = path
                hamilton_status = "存在"
            else:
                hamilton_status = f"存在，但缺少边: {missing_edges}" if missing_edges else "未覆盖所有景点"
        else:
            hamilton_status = "不存在"
    except Exception as e:
        hamilton_status = f"求解失败: {str(e)}"

    euler_path = None
    euler_time = float("inf")
    euler_status = "无解"

    return hamilton_path, hamilton_time, hamilton_status, euler_path, euler_time, euler_status, filtered_df, simple_filtered_df

def plan_paths_simu(selected_nodes, df, simple_df, edges=None):
    # 路径过滤
    filtered_df = df[df["起点"].isin(selected_nodes) & df["终点"].isin(selected_nodes)]
    filtered_df = filtered_df[['起点', '终点', '预计步行时间_分钟']]
    filtered_df["起点"] = filtered_df["起点"].str.strip()
    filtered_df["终点"] = filtered_df["终点"].str.strip()
    
    simple_filtered_df = simple_df[simple_df["起点"].isin(selected_nodes) & simple_df["终点"].isin(selected_nodes)]
    simple_filtered_df = simple_filtered_df[['起点', '终点', '预计步行时间_分钟']]

    # 检查数据完整性
    missing_nodes = [node for node in selected_nodes if node not in filtered_df["起点"].values and node not in filtered_df["终点"].values]
    if missing_nodes:
        st.error(f"❌ 数据中缺少以下景点的路径信息：{missing_nodes}")
        return None, 0, f"数据缺失: {missing_nodes}", None, 0, "无解", filtered_df, simple_filtered_df

    # 创建边列表
    edges = list(filtered_df.itertuples(index=False, name=None))  
    edge_pairs = [(u, v, w) for u, v, w in edges if u != v] + [(v, u, w) for u, v, w in edges if u != v]

    simple_edges = list(simple_filtered_df.itertuples(index=False, name=None))  
    simple_edge_pairs = [(u, v, w) for u, v, w in simple_edges if u != v] + [(v, u, w) for u, v, w in simple_edges if u != v]
    

    if len(simple_edge_pairs) < len(selected_nodes) - 1 or len(edge_pairs) < len(selected_nodes) - 1:
        st.warning("子图边数不足，可能无法形成连通图！")
        return None, 0, "边数不足", None, 0, "无解", filtered_df, simple_filtered_df

    # 检查有向图连通性
    G_temp = nx.DiGraph()
    G_temp.add_weighted_edges_from(edge_pairs)
    G_undirected = G_temp.to_undirected()
    if not nx.is_connected(G_undirected):
        st.error("❌ 子图不连通，无法覆盖所有景点！")
        return None, 0, "子图不连通", None, 0, "无解", filtered_df, simple_filtered_df

    ecp = EulerProgramming()
    ecp.addEdges(simple_edge_pairs)
    hmp = HamiltonProgramming()
    hmp.addEdges(edge_pairs)

    hamilton_path = None
    hamilton_time = float("inf")
    hamilton_status = "无解"

    try:
        hmp.HamiltonPath_solver()
        if hasattr(hmp, "hamiltonian_path") and hmp.hamiltonian_path:
            path = hmp.hamiltonian_path
            hamilton_time = 0
            missing_edges = []
            for u, v in path:
                sub = filtered_df[((filtered_df["起点"] == u) & (filtered_df["终点"] ==v)) |
                                        ((filtered_df["起点"] ==v) & (filtered_df["终点"] == u))]
                if not sub.empty:
                    hamilton_time += sub["预计步行时间_分钟"].min()
                else:
                    missing_edges.append((u, v))
            if not missing_edges and len(set(u for u, _ in path).union({path[-1][1]})) == len(selected_nodes):
                hamilton_path = path
                hamilton_status = "存在"
            else:
                hamilton_status = f"存在，但缺少边: {missing_edges}" if missing_edges else "未覆盖所有景点"
        else:
            hamilton_status = "不存在"
    except Exception as e:
        hamilton_status = f"求解失败: {str(e)}"

    euler_path = None
    euler_time = float("inf")
    euler_status = "无解"

    try:
        ecp.EulerPath_solver()
        if hasattr(ecp, "eulerian_path") and ecp.eulerian_path:
            path = ecp.eulerian_path
            euler_time = 0
            missing_edges = []
            for u, v in path:
                sub = simple_filtered_df[((simple_filtered_df["起点"] == str(u)) & (simple_filtered_df["终点"] == str(v))) |
                                        ((simple_filtered_df["起点"] == str(v)) & (simple_filtered_df["终点"] == str(u)))]
                if not sub.empty:
                    euler_time += sub["预计步行时间_分钟"].min()
                else:
                    missing_edges.append((u, v))
            if not missing_edges and len(set(u for u, _ in path).union({path[-1][1]})) == len(selected_nodes):
                euler_path = path
                euler_status = "存在"
            else:
                euler_status = f"存在，但缺少边: {missing_edges}" if missing_edges else "未覆盖所有景点"
        else:
            euler_status = "不存在"
    except Exception as e:
        euler_status = f"求解失败: {str(e)}"

    return hamilton_path, hamilton_time, hamilton_status, euler_path, euler_time, euler_status, filtered_df, simple_filtered_df
