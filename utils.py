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
from Render import display_path_comparison_map,display_path_comparison_map_simu
from path_plan import plan_paths,plan_paths_simu
from Hamilton import HamiltonProgramming
from Euler import EulerProgramming


def handle_path_planning(selected_spots, df, simple_df,locations_df, node_coordinates_df,center, boundary_coords, max_points, is_shortest_only):
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

    display_path_results(results,df, locations_df, node_coordinates_df,center, boundary_coords, max_points, is_shortest_only=is_shortest_only)


def display_path_results(results, df, locations_df, node_coordinates_df,center, boundary_coords, max_points, is_shortest_only=False):
    h_path, h_time, h_status, e_path, e_time, e_status, *_ = results

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

    display_map(df, locations_df, node_coordinates_df,center, boundary_coords, max_points,is_shortest_only)


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
    if h_time and e_time:
        if h_time < e_time:
            st.success(f"🏆 哈密顿路径更优，预计 {h_time:.1f} 分钟")
        elif e_time < h_time:
            st.success(f"🏆 欧拉路径更优，预计 {e_time:.1f} 分钟")
        else:
            st.info(f"⚖️ 两种路径时间相等，均为 {h_time:.1f} 分钟")
    else:
        st.error("❌ 无可行路径")


def display_map(
    df, locations_df, node_coordinates_df,
    center, boundary_coords, max_points,is_shortest_only=True
):
    if st.session_state.selected_spots and st.session_state.path_results:
        if is_shortest_only:
            # 如果是安全巡视，调用支持non_tourism参数的函数
            display_path_comparison_map(
                st.session_state.selected_spots,
                st.session_state.path_results,
                df,
                locations_df,
                node_coordinates_df,
                center,
                boundary_coords,
                max_points
            )
        else:
            # 否则调用不支持non_tourism参数的函数
            display_path_comparison_map_simu(
                st.session_state.selected_spots,
                st.session_state.path_results,
                df,
                locations_df,
                node_coordinates_df,
                center,
                boundary_coords,
                max_points
            )

