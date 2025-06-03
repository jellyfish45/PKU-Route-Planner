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

def get_full_path_coordinates(path, path_detail_df, node_id_to_coord, max_points=200):
        if not path:
            return []
        full_node_list = []
        for u, v in path:
            match = path_detail_df[
                ((path_detail_df["起点"] == u) & (path_detail_df["终点"] == v)) |
                ((path_detail_df["起点"] == v) & (path_detail_df["终点"] == u))
            ]
            if not match.empty:
                best_row = match.sort_values("预计步行时间_分钟").iloc[0]
                node_ids = best_row["路径节点"]
                if full_node_list and node_ids and full_node_list[-1] == node_ids[0]:
                    node_ids = node_ids[1:]
                full_node_list.extend(node_ids)
        if len(full_node_list) > max_points:
            indices = np.linspace(0, len(full_node_list) - 1, max_points, dtype=int)
            full_node_list = [full_node_list[i] for i in indices]
        coords = [node_id_to_coord[node] for node in full_node_list if node in node_id_to_coord]
        return coords

def display_path_comparison_map(
    recommended_spots: list[str],
    path_results: dict,
    df: pd.DataFrame,
    locations_df: pd.DataFrame,
    node_coordinates_df: pd.DataFrame,
    center: list[float],
    boundary_coords: list[list[float]],
    max_points: int = 200,
):
   
    filtered_df = path_results["filtered_df"]
    simple_filtered_df = path_results["simple_filtered_df"]

    # 调试信息：检查路径
    hamilton_path = path_results["hamilton_path"]
    hamilton_time = path_results["hamilton_time"]
    hamilton_status = path_results["hamilton_status"]
    st.write(f"哈密顿通路: {hamilton_path}")
    hamilton_map = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap", control_scale=True)
    # 添加校园边界
    folium.PolyLine(boundary_coords, color="black", weight=2.5, opacity=0.8).add_to(hamilton_map)


    # 添加标记（根据传入的景点）
    for _, row in locations_df[locations_df["名称"].isin(recommended_spots)].iterrows():
        folium.Marker(
            location=[row["纬度"], row["经度"]],
            tooltip=row["名称"],
            popup=row["名称"],
            icon=folium.Icon(color="green")
        ).add_to(hamilton_map)

    # 路径坐标生成辅助函数
    df["路径节点"] = df["路径节点"].apply(ast.literal_eval)

    node_id_to_coord = dict(zip(node_coordinates_df["node"], zip(node_coordinates_df["lat"], node_coordinates_df["lng"])))

    # 绘制哈密顿通路（绿色虚线）
    if hamilton_path:
        hamilton_coords = get_full_path_coordinates(hamilton_path, df, node_id_to_coord, max_points=max_points)
        if hamilton_coords:
            st.write(f"哈密顿通路坐标点数: {len(hamilton_coords)}")
            folium.PolyLine(
                hamilton_coords,
                color="green",
                weight=5,
                opacity=0.8,
                dash_array="5, 5",
                tooltip=f"哈密顿通路，预计 {hamilton_time:.1f} 分钟"
            ).add_to(hamilton_map)
        else:
            st.warning("哈密顿通路坐标为空，可能数据问题！")

        folium.LayerControl().add_to(hamilton_map)
    
        # 显示哈密顿地图
        st.subheader("游览路径规划（哈密顿通路）")
        st_folium(hamilton_map, width=650, height=450, key=f"hamilton_map_{len(recommended_spots)}", returned_objects=[])


def display_path_comparison_map_simu(
    recommended_spots: list[str],
    path_results: dict,
    df: pd.DataFrame,
    locations_df: pd.DataFrame,
    node_coordinates_df: pd.DataFrame,
    center: list[float],
    boundary_coords: list[list[float]],
    max_points: int = 200
):

    hamilton_path = path_results["hamilton_path"]
    hamilton_time = path_results["hamilton_time"]
    hamilton_status = path_results["hamilton_status"]
    euler_path = path_results["euler_path"]
    euler_time = path_results["euler_time"]
    euler_status = path_results["euler_status"]
    filtered_df = path_results["filtered_df"]
    simple_filtered_df = path_results["simple_filtered_df"]

    # 调试信息：检查路径
    st.write(f"哈密顿路径: {hamilton_path}")
    st.write(f"欧拉路径: {euler_path}")

    # 初始化两个独立的地图
    hamilton_map = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap", control_scale=True)
    euler_map = folium.Map(location=center, zoom_start=17, tiles="OpenStreetMap", control_scale=True)

    # 添加校园边界
    folium.PolyLine(boundary_coords, color="black", weight=2.5, opacity=0.8).add_to(hamilton_map)
    folium.PolyLine(boundary_coords, color="black", weight=2.5, opacity=0.8).add_to(euler_map)

    # 添加标记（分别添加到两个地图）
    for _, row in locations_df[locations_df["名称"].isin(recommended_spots)].iterrows():
        # 哈密顿地图的标记
        folium.Marker(
            location=[row["纬度"], row["经度"]],
            tooltip=row["名称"],
            popup=row["名称"],
            icon=folium.Icon(color="green")
        ).add_to(hamilton_map)
        # 欧拉地图的标记
        folium.Marker(
            location=[row["纬度"], row["经度"]],
            tooltip=row["名称"],
            popup=row["名称"],
            icon=folium.Icon(color="green")
        ).add_to(euler_map)

    # 路径坐标生成
    df["路径节点"] = df["路径节点"].apply(ast.literal_eval)
    
    node_id_to_coord = dict(zip(node_coordinates_df["node"], zip(node_coordinates_df["lat"], node_coordinates_df["lng"])))

    # 绘制哈密顿路径（绿色虚线）
    if hamilton_path:
        hamilton_coords = get_full_path_coordinates(hamilton_path, df, node_id_to_coord, max_points=max_points)
        if hamilton_coords:
            st.write(f"哈密顿路径坐标点数: {len(hamilton_coords)}")
            folium.PolyLine(
                hamilton_coords,
                color="green",
                weight=5,
                opacity=0.8,
                dash_array="5, 5",
                tooltip=f"哈密顿路径，预计 {hamilton_time:.1f} 分钟"
            ).add_to(hamilton_map)
        else:
            st.warning("哈密顿路径坐标为空，可能数据问题！")

    # 绘制欧拉路径（红色实线）
    if euler_path:
        euler_coords = get_full_path_coordinates(euler_path, df, node_id_to_coord, max_points=max_points)
        if euler_coords:
            st.write(f"欧拉路径坐标点数: {len(euler_coords)}")
            folium.PolyLine(
                euler_coords,
                color="red",
                weight=5,
                opacity=0.8,
                tooltip=f"欧拉路径，预计 {euler_time:.1f} 分钟"
            ).add_to(euler_map)
        else:
            st.warning("欧拉路径坐标为空，可能数据问题！")

    # 添加图层控制
    folium.LayerControl().add_to(hamilton_map)
    folium.LayerControl().add_to(euler_map)

    # 显示地图
    st.subheader("路径对比地图")
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("### 哈密顿路径")
        st_folium(hamilton_map, width=650, height=450, key=f"hamilton_map_{len(recommended_spots)}", returned_objects=[])
    with col2:
        st.markdown("### 欧拉路径")
        st_folium(euler_map, width=650, height=450, key=f"euler_map_{len(recommended_spots)}", returned_objects=[])
