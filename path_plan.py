import pandas as pd
import streamlit as st
from Hamilton import HamiltonProgramming
from Euler import EulerProgramming

def plan_paths(selected_nodes, df, simple_df):
    # 路径过滤
    filtered_df = df[df["起点"].isin(selected_nodes) & df["终点"].isin(selected_nodes)]
    filtered_df = filtered_df[['起点', '终点', '预计步行时间_分钟']]
    simple_filtered_df = simple_df[simple_df["起点"].isin(selected_nodes) & simple_df["终点"].isin(selected_nodes)]
    simple_filtered_df = simple_filtered_df[['起点', '终点', '预计步行时间_分钟']]

    # 限制子图规模
    if len(filtered_df) > 16:
        filtered_df = filtered_df.nsmallest(16, '预计步行时间_分钟')

    # 创建边列表
    edges = list(filtered_df.itertuples(index=False, name=None))
    edge_pairs=[]
    for u, v, w in edges:
        edge_pairs.append((u, v, w))
        edge_pairs.append((v, u, w)) 

    if len(edge_pairs) < 2:
        st.warning("子图边数过少，可能无解！尝试选择更多景点或调整过滤条件。")

    hmp = HamiltonProgramming()
    hmp.addEdges(edge_pairs)

    hamilton_path = None
    hamilton_time = float("inf")
    hamilton_status = "无解"

    try:
        hmp.HamiltonPath_solver()
        if hasattr(hmp, "hamiltonian_path"):
            path = hmp.hamiltonian_path
            hamilton_time = 0
            missing_edges = []
            for u, v in path:
                sub = filtered_df[((filtered_df["起点"] == str(u)) & (filtered_df["终点"] == str(v))) |
                                         ((filtered_df["起点"] == str(v)) & (filtered_df["终点"] == str(u)))]
                if not sub.empty:
                    hamilton_time += sub["预计步行时间_分钟"].min()
                else:
                    missing_edges.append((u, v))
            if not missing_edges:
                hamilton_path = path
                hamilton_status = "存在"
            else:
                hamilton_status = f"存在，但缺少边: {missing_edges}"
        else:
            hamilton_status = "不存在"
    except Exception as e:
        hamilton_status = f"求解失败: {str(e)}"

    euler_path = None
    euler_time = float("inf")
    euler_status = "无解"


    return hamilton_path, hamilton_time, hamilton_status, euler_path, euler_time, euler_status, filtered_df, simple_filtered_df


def plan_paths_simu(selected_nodes, df, simple_df):
    # 路径过滤
    filtered_df = df[df["起点"].isin(selected_nodes) & df["终点"].isin(selected_nodes)]
    filtered_df = filtered_df[['起点', '终点', '预计步行时间_分钟']]
    simple_filtered_df = simple_df[simple_df["起点"].isin(selected_nodes) & simple_df["终点"].isin(selected_nodes)]
    simple_filtered_df = simple_filtered_df[['起点', '终点', '预计步行时间_分钟']]

    # 限制子图规模
    if len(filtered_df) > 16:
        filtered_df = filtered_df.nsmallest(16, '预计步行时间_分钟')
    if len(simple_filtered_df) > 16:
        simple_filtered_df = simple_filtered_df.nsmallest(16, '预计步行时间_分钟')

    # 创建边列表
    edges = list(filtered_df.itertuples(index=False, name=None))
    edge_pairs = [(u, v) for u, v, _ in edges]
    simple_edges = list(simple_filtered_df.itertuples(index=False, name=None))
    simple_edge_pairs = [(u, v) for u, v, _ in simple_edges]

    if len(simple_edge_pairs) < 2 or len(edge_pairs) < 2:
        st.warning("子图边数过少，可能无解！尝试选择更多景点或调整过滤条件。")

    ecp = EulerProgramming()
    ecp.addEdges(simple_edge_pairs)
    hmp = HamiltonProgramming()
    hmp.addEdges(edge_pairs)

    hamilton_path = None
    hamilton_time = float("inf")
    hamilton_status = "无解"

    try:
        hmp.HamiltonPath_solver()
        if hasattr(hmp, "hamiltonian_path"):
            path = hmp.hamiltonian_path
            hamilton_time = 0
            missing_edges = []
            for u, v in path:
                sub = filtered_df[((filtered_df["起点"] == str(u)) & (filtered_df["终点"] == str(v))) |
                                         ((filtered_df["起点"] == str(v)) & (filtered_df["终点"] == str(u)))]
                if not sub.empty:
                    hamilton_time += sub["预计步行时间_分钟"].min()
                else:
                    missing_edges.append((u, v))
            if not missing_edges:
                hamilton_path = path
                hamilton_status = "存在"
            else:
                hamilton_status = f"存在，但缺少边: {missing_edges}"
        else:
            hamilton_status = "不存在"
    except Exception as e:
        hamilton_status = f"求解失败: {str(e)}"

    euler_path = None
    euler_time = float("inf")
    euler_status = "无解"

    try:
        ecp.EulerPath_solver()
        if hasattr(ecp, "eulerian_path"):
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
            if not missing_edges:
                euler_path = path
                euler_status = "存在"
            else:
                euler_status = f"存在，但缺少边: {missing_edges}"
        else:
            euler_status = "不存在"
    except Exception as e:
        euler_status = f"求解失败: {str(e)}"

    return hamilton_path, hamilton_time, hamilton_status, euler_path, euler_time, euler_status, filtered_df, simple_filtered_df
