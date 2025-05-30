import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
import folium
from streamlit_folium import st_folium
import ast
from Hamilton import HamiltonProgramming
from Euler import EulerProgramming

#1. 读入数据
df=pd.read_csv("./data/pku_all_simple_paths_15.csv")
Simple_df=df.copy()

df=df[(df['预计步行时间_分钟']<=8.4)&(df['预计步行时间_分钟']>=6.6)] #简单数据筛选一下，不然数据量太大，易达到python极限

Simple_df["无向边"] = Simple_df.apply(lambda row: tuple(sorted([row["起点"], row["终点"]])), axis=1)
Simple_df_cleaned = Simple_df.loc[Simple_df.groupby("无向边")["预计步行时间_分钟"].idxmin()].reset_index(drop=True)
simple_df=Simple_df_cleaned.drop(columns=["无向边"])


locations_df = pd.read_csv("./data/pku_locations_updated.csv")  
locations_df["经度"] = locations_df["经纬度"].apply(lambda x: float(x.split(",")[0]))
locations_df["纬度"] = locations_df["经纬度"].apply(lambda x: float(x.split(",")[1]))

locations_df[["lng", "lat"]] = locations_df["经纬度"].str.split(",", expand=True).astype(float)
restricted = {"桥", "椅", "卫生间", "文创"}
condition = (~locations_df["名称"].str.extract(f"({'|'.join(restricted)})")[0].notna()) | (locations_df["是否在校内"] == "校内")
locations_df = locations_df[condition].copy()

#2. Streamlit 页面布局 
st.title("北京大学游览路径规划器")
st.markdown("请选择要连接的若干地点，判断是否可构成通路，并找出用时最短的路径")
center = [locations_df["lat"].mean(), locations_df["lng"].mean()]
m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap")

#3. 地图与边界 
boundary_coords = [
    (39.9851791, 116.3004617), (39.9864006, 116.3003677), (39.9863900, 116.2994966),
    (39.9931543, 116.2984634), (39.9931592, 116.2985974), (39.9933919, 116.2985916),
    (39.9933893, 116.2984734), (39.9939562, 116.2984578), (39.9948007, 116.2984264),
    (39.9958545, 116.2983911), (39.9961909, 116.3008489), (39.9962501, 116.3011623),
    (39.9963362, 116.3013332), (39.9965626, 116.3015876), (39.9968540, 116.3018948),
    (39.9973739, 116.3024428), (39.9975695, 116.3028049), (39.9977573, 116.3032563),
    (39.9978738, 116.3038116), (39.9982026, 116.3053797), (39.9982346, 116.3056759),
    (39.9982373, 116.3058293), (39.9982402, 116.3059979), (39.9970351, 116.3062102),
    (39.9970828, 116.3067154), (39.9957638, 116.3069369), (39.9959020, 116.3089361),
    (39.9929106, 116.3091399), (39.9907323, 116.3093135), (39.9906958, 116.3092952),
    (39.9888646, 116.3094595), (39.9887974, 116.3094367), (39.9887223, 116.3094698),
    (39.9879948, 116.3095240), (39.9879245, 116.3095036), (39.9878347, 116.3095240),
    (39.9868598, 116.3095834), (39.9867679, 116.3095553), (39.9866939, 116.3095894),
    (39.9864591, 116.3095878), (39.9860867, 116.3095845), (39.9857161, 116.3093521),
    (39.9853308, 116.3093613), (39.9853151, 116.3085630), (39.9852700, 116.3054565),
    (39.9851791, 116.3004617)
]
folium.PolyLine(boundary_coords, color="black", weight=2.5).add_to(m)

#4. 添加标记
marker_dict = {}

for _, row in locations_df.iterrows():
    marker = folium.Marker(
        location=[row["纬度"], row["经度"]],
        tooltip=row["名称"],
        popup=row["名称"],
    )
    marker.add_to(m)
    marker_dict[row["名称"]] = (row["纬度"], row["经度"])

#5. 用于记录用户选中的节点
selected_nodes = st.multiselect("请选择节点：", options=locations_df["名称"].tolist())


#7. 判断
best_path=[]
best_type=None
best_time=None
if st.button("判断是否存在通路"):
    if len(selected_nodes) < 2:
        st.warning("请至少选择两个节点")
    else:
        Filtered_df = df[df["起点"].isin(selected_nodes) & df["终点"].isin(selected_nodes)]
        filtered_df = Filtered_df[['起点', '终点', '预计步行时间_分钟']]

        simple_Filtered_df = simple_df[simple_df["起点"].isin(selected_nodes) & simple_df["终点"].isin(selected_nodes)]
        simple_filtered_df = simple_Filtered_df[['起点', '终点', '预计步行时间_分钟']]
        
        edges = list(filtered_df.itertuples(index=False, name=None))
        edge_pairs = [(u, v) for u, v, _ in edges]

        simple_edges = list(simple_filtered_df.itertuples(index=False, name=None))
        simple_edge_pairs = [(u, v) for u, v, _ in simple_edges]

        ecp = EulerProgramming()
        ecp.addEdges(edge_pairs)
        
        hmp = HamiltonProgramming()
        hmp.addEdges(simple_edge_pairs)

        best_path = None
        best_time = float("inf")
        best_type = None

        # 哈密顿通路
        with st.spinner("Gurobi 正在判断哈密顿通路..."):
            hmp.HamiltonPath_solver()
            if hasattr(hmp, "hamiltonian_path"):
                path = hmp.hamiltonian_path
                hamilton_time = 0
                missing_edges = []

                for u, v in path:
                    sub = simple_filtered_df[((simple_filtered_df["起点"] == str(u)) & (simple_filtered_df["终点"] == str(v))) |
                                      ((simple_filtered_df["起点"] == str(v)) & (simple_filtered_df["终点"] == str(u)))]
                    if not sub.empty:
                        hamilton_time += sub["预计步行时间_分钟"].min()
                    else:
                        missing_edges.append((u, v))

                if not missing_edges:
                    st.success("✅ 数学规划判断：存在哈密顿通路")
                    best_path = path
                    best_time = hamilton_time
                    best_type = "哈密顿通路"
                else:
                    st.warning(f"⚠️ 哈密顿通路存在，但以下边无预计时间数据: {missing_edges}")
            else:
                st.error("❌ 数学规划判断：不存在哈密顿通路")

        # 欧拉路径
        with st.spinner("Gurobi 正在判断欧拉路径..."):
            ecp.EulerPath_solver()
            if hasattr(ecp, "eulerian_path"):
                path = ecp.eulerian_path
                euler_time = 0
                missing_edges = []

                for u, v in path:
                    sub = filtered_df[((filtered_df["起点"] == str(u)) & (filtered_df["终点"] == str(v))) |
                                      ((filtered_df["起点"] == str(v)) & (filtered_df["终点"] == str(u)))]
                    if not sub.empty:
                        euler_time += sub["预计步行时间_分钟"].min()
                    else:
                        missing_edges.append((u, v))

                if not missing_edges:
                    st.success("✅ 数学规划判断：存在欧拉路径")
                    if euler_time < best_time:
                        best_path = path
                        best_time = euler_time
                        best_type = "欧拉路径"
                else:
                    st.warning(f"⚠️ 欧拉路径存在，但以下边无预计时间数据: {missing_edges}")
            else:
                st.error("❌ 数学规划判断：不存在欧拉路径")

        # 结果输出与持久化
        if best_path is not None:
            st.session_state.best_path = best_path
            st.session_state.best_time = best_time
            st.session_state.best_type = best_type
        else:
            st.error("❌ 无可行路径")

# 8. 持久化结果展示
if "best_path" in st.session_state:
    best_path = st.session_state.best_path
    best_time = st.session_state.best_time
    best_type = st.session_state.best_type

    st.markdown(f"🏆 **更优方案为：{best_type}**")
    st.markdown(f"🔀 路径：{' -> '.join([str(u) for u, v in best_path] + [str(best_path[-1][1])])}")
    st.markdown(f"🕒 预计最短步行时间：{best_time:.1f} 分钟")

    df["路径节点"] = df["路径节点"].apply(ast.literal_eval)
    
    def get_full_path_coordinates(best_path, path_detail_df, node_id_to_coord):
        full_node_list = []
        for u, v in best_path:
            match = path_detail_df[
                ((path_detail_df["起点"] == u) & (path_detail_df["终点"] == v)) |
                ((path_detail_df["起点"] == v) & (path_detail_df["终点"] == u))
            ]
            if not match.empty:
                best_row = match.sort_values("预计步行时间_分钟").iloc[0]
                node_ids = best_row["路径节点"]
                if node_ids[0] != node_ids[-1] and full_node_list and full_node_list[-1] == node_ids[0]:
                    node_ids = node_ids[1:]  # 避免重复
                full_node_list.extend(node_ids)
        return [node_id_to_coord[node] for node in full_node_list if node in node_id_to_coord]
    
    node_coordinates_df = pd.read_csv("./data/pku_walk_node_locations.csv")  
    node_id_to_coord = dict(zip(node_coordinates_df["node"], zip(node_coordinates_df["lat"], node_coordinates_df["lng"])))
    
    path_coords = get_full_path_coordinates(best_path, df, node_id_to_coord)
    
    if path_coords:
        folium.PolyLine(
            path_coords,
            color="red",
            weight=5,
            opacity=0.8,
            tooltip=f"{best_type}，预计 {best_time:.1f} 分钟"
        ).add_to(m)


#6. 显示地图
st_data = st_folium(m, width=700, height=500)
