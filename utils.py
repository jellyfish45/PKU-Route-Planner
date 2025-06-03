import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
import folium
from streamlit_folium import st_folium
import ast
from Hamilton import HamiltonProgramming
from Euler import EulerProgramming
import sys
from time import sleep
import os
from tqdm import tqdm
import re
import json
import concurrent.futures
from openai import OpenAI
from Render import display_path_comparison_map, display_path_comparison_map_simu
from path_plan import plan_paths, plan_paths_simu

def handle_path_planning(selected_spots, df, simple_df, locations_df, node_coordinates_df, center, boundary_coords, max_points, is_shortest_only):
    # 检查 selected_spots 是否有效
    if len(selected_spots) < 2:
        st.error("❌ 请至少选择2个景点！")
        return
    if len(selected_spots) > 8:
        st.error("❌ 最多选择8个景点！")
        return

    # 检查数据完整性
    missing_nodes = [spot for spot in selected_spots if spot not in simple_df["起点"].values and spot not in simple_df["终点"].values]
    if missing_nodes:
        st.error(f"❌ 数据中缺少以下景点的路径信息：{missing_nodes}")
        return

    # 调用路径规划函数
    if is_shortest_only:
        results = plan_paths(selected_spots, df, simple_df)
    else:
        results = plan_paths_simu(selected_spots, df, simple_df)

    st.session_state.selected_spots = selected_spots
    st.session_state.path_results = {
        "hamilton_path": results[0],
        "hamilton_time": results[1],
        "hamilton_status": results[2],
        "euler_path": results[3],
        "euler_time": results[4],
        "euler_status": results[5],
        "filtered_df": results[6],
        "simple_filtered_df": results[7]
    }

    display_path_results(results, df, locations_df, node_coordinates_df, center, boundary_coords, max_points, is_shortest_only=is_shortest_only)

def display_path_results(results, df, locations_df, node_coordinates_df, center, boundary_coords, max_points, is_shortest_only=False):
    h_path, h_time, h_status, e_path, e_time, e_status, filtered_df, simple_filtered_df = results

    # 调试：打印路径权重
    if h_path:
        weights = []
        for u, v in h_path:
            sub = simple_filtered_df[((simple_filtered_df['起点'] == str(u)) & (simple_filtered_df['终点'] == str(v))) |
                                    ((simple_filtered_df['起点'] == str(v)) & (simple_filtered_df['终点'] == str(u)))]
            weights.append(sub["预计步行时间_分钟"].min() if not sub.empty else "缺失")
        #st.write(f"[调试] 哈密顿路径边权重：{weights}")
    if e_path and not is_shortest_only:
        weights = []
        for u, v in e_path:
            sub = simple_filtered_df[((simple_filtered_df['起点'] == str(u)) & (simple_filtered_df['终点'] == str(v))) |
                                    ((simple_filtered_df['起点'] == str(v)) & (simple_filtered_df['终点'] == str(u)))]
            weights.append(sub["预计步行时间_分钟"].min() if not sub.empty else "缺失")
        #st.write(f"[调试] 欧拉路径边权重：{weights}")

    if is_shortest_only:
        st.markdown("### 哈密顿路径")
        display_path("哈密顿", h_path, h_time, h_status)
        st.markdown("**图例**: 哈密顿通路（绿色虚线），景点（绿色标记），校园边界（黑色）")
    else:
        st.markdown("### 哈密顿路径")
        display_path("哈密顿", h_path, h_time, h_status)
        st.markdown("### 欧拉路径")
        display_path("欧拉", e_path, e_time, e_status)
        compare_paths(h_time, e_time)

    display_map(df, locations_df, node_coordinates_df, center, boundary_coords, max_points, is_shortest_only)

def display_path(name, path, time, status):
    st.markdown(f"**状态**: {status}")
    if path:
        path_str = " -> ".join([str(u) for u, v in path] + [str(path[-1][1])])
        st.markdown(f"**路径**: {path_str}")
        st.markdown(f"**预计步行时间**: {time:.1f} 分钟")
    else:
        st.markdown("**路径**: 无")
        st.markdown("**预计步行时间**: 无")

def compare_paths(h_time, e_time):
    if h_time != float('inf') and e_time != float('inf'):
        if h_time < e_time:
            st.success(f"🏆 哈密顿路径 [{h_time:.1f} 分钟] 优于欧拉路径 [{e_time:.1f} 分钟]")
        elif h_time > e_time:
            st.success(f"🏆 欧拉路径 [{e_time:.1f} 分钟] 优于哈密顿路径 [{h_time:.1f} 分钟]")
        else:
            st.info(f"⚖️ 哈密顿路径和欧拉路径时间相同：{h_time:.1f} 分钟")
    else:
        st.warning("⚠️ 无法比较路径：至少一个路径无效")

def display_map(
    df, locations_df, node_coordinates_df,
    center, boundary_coords, max_points, is_shortest_only=True
):
    if st.session_state.get('selected_spots') and st.session_state.get('path_results'):
        if is_shortest_only:
            display_path_comparison_map(
                st.session_state['selected_spots'],
                st.session_state['path_results'],
                df,
                locations_df,
                node_coordinates_df,
                center,
                boundary_coords,
                max_points
            )
        else:
            display_path_comparison_map_simu(
                st.session_state['selected_spots'],
                st.session_state['path_results'],
                df,
                locations_df,
                node_coordinates_df,
                center,
                boundary_coords,
                max_points
            )
