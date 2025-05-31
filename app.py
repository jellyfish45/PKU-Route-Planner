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
import os
from time import sleep
from tqdm import tqdm
import re
import json
import concurrent.futures
from openai import OpenAI

# 增加递归深度限制（临时方案）
sys.setrecursionlimit(2000)

# OpenAI API 配置
base_url = "http://123.129.219.111:3000/v1"
api_key = "sk-J4OU0nswdAQEmN7y7pS9ytPedSvEC8NXCOhuBX5GIz3dXz3c"
max_try_num = 5
max_thread_num = 8
parse_json = True
gpt_model = "gpt-4o"
max_tokens = 4096
temperature = 0.5

# 景点列表
spot_list = ['三一八烈士纪念碑', '临湖轩', '乾隆半月台诗碑', '乾隆诗碑', '华表', '博雅塔', '卫生间', '图书馆', '埃德加·斯诺之墓', 
             '塞万提斯像', '塞克勒博物馆', '大水法石构件', '慈济寺山门', '文创', '断桥残雪石坊楣', '新太阳', '日晷', '朗润园', 
             '未名湖口', '未名湖石', '李大钊像', '柳浪闻莺石坊楣', '校史馆', '校景亭', '植树碑', '燕南园口', '百周年纪念讲堂', 
             '石舫', '石雕屏风', '红湖口', '翻尾石鱼', '花神庙', '茜园梅石碑', '荷花池口', '葛利普教授之墓', '蔡元培像', 
             '西南联大纪念碑', '西门', '赖朴吾和夏仁德墓', '钟亭', '镜春园', '静园', '鲁斯亭']

# 推荐模板
template = """
你是一名北京大学的导游，北京大学校内包含以下景点。\n\n{spot_list}
请根据以下游客需求给出推荐的5-10个景点序列，返回一个json列表，不要返回其他内容。
游客描述:\n\n{text}
"""

# JSON 解析类
class ParseJson:
    def __init__(self):
        super().__init__()

    def replace_newlines(self, match):
        return match.group(0).replace('\n', '\\n').replace('\r', '\\r')

    def clean_json_str(self, json_str: str) -> str:
        json_str = json_str.replace("None", "null")
        match = re.search(r'```json(.*?)```', json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        match = re.search(r'```(.*?)```', json_str, re.DOTALL)
        if match:
            json_str = match.group(1)
        json_str = re.sub(r'("(?:\\.|[^"\\])*")', self.replace_newlines, json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r'\"\s+\"', '\",\"', json_str)
        json_str = json_str.replace("True", "true").replace("False", "false")
        return json_str

    def txt2obj(self, text):
        try:
            text = self.clean_json_str(text)
            return json.loads(text)
        except Exception as e:
            print(e)
            return None

# ChatCompletion 类
class ChatCompletion:
    def __init__(self, chunks):
        self.chunks = chunks
        self.template = template
        if parse_json:
            self.parse_json = ParseJson()

    def _get_chat_completion(self, chunk):
        messages = [{"role": "user", "content": template.format(spot_list=spot_list, text=chunk)}]
        client = OpenAI(api_key=api_key, base_url=base_url)
        chat_completion = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0
        )
        if parse_json:
            return self.parse_json.txt2obj(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content

    def get_chat_completion(self, chunk):
        retry = 0
        while retry < max_try_num:
            try:
                return self._get_chat_completion(chunk)
            except Exception as e:
                retry += 1
                sleep(0.1 * retry)
                print(e)
        else:
            raise Exception("Max try number reached.")
            return None

    def complete(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_num) as executor:
            future_results = list(tqdm(executor.map(self.get_chat_completion, self.chunks), total=len(self.chunks)))
        return future_results

# 数据加载（缓存优化）
@st.cache_data
def load_data():
    df = pd.read_csv("./data/pku_all_simple_paths_15.csv")
    locations_df = pd.read_csv("./data/pku_locations_updated.csv")
    node_coordinates_df = pd.read_csv("./data/pku_walk_node_locations.csv")
    
    # 路径过滤
    df = df[(df['预计步行时间_分钟'] <= 8.4) & (df['预计步行时间_分钟'] >= 6.6)]
    Simple_df = df.copy()
    Simple_df["无向边"] = Simple_df.apply(lambda row: tuple(sorted([row["起点"], row["终点"]])), axis=1)
    Simple_df_cleaned = Simple_df.loc[Simple_df.groupby("无向边")["预计步行时间_分钟"].idxmin()].reset_index(drop=True)
    simple_df = Simple_df_cleaned.drop(columns=["无向边"])
    
    # 地点数据处理
    locations_df["经度"] = locations_df["经纬度"].apply(lambda x: float(x.split(",")[0]))
    locations_df["纬度"] = locations_df["经纬度"].apply(lambda x: float(x.split(",")[1]))
    locations_df[["lng", "lat"]] = locations_df["经纬度"].str.split(",", expand=True).astype(float)
    restricted = {"桥", "椅", "卫生间", "文创"}
    condition = (~locations_df["名称"].str.extract(f"({'|'.join(restricted)})")[0].notna()) | (locations_df["是否在校内"] == "校内")
    locations_df = locations_df[condition].copy()
    
    return df, simple_df, locations_df, node_coordinates_df

# 加载数据
df, simple_df, locations_df, node_coordinates_df = load_data()

# 校园边界坐标
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

# Streamlit 页面布局
st.title("北京大学游览路径规划器")
st.markdown("请输入您的游览需求（如‘我喜欢去在校学生多的地方’），系统将推荐景点并规划最短路径。")

# 缓存地图初始化（恢复 OpenStreetMap 瓦片）
@st.cache_resource
def init_map(center):
    m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap", control_scale=True)
    return m

# 初始化地图（仅边界）
center = [locations_df["lat"].mean(), locations_df["lng"].mean()]
m = init_map(center)
folium.PolyLine(boundary_coords, color="black", weight=2.5, opacity=0.8).add_to(m)

# 缓存API调用结果
@st.cache_data
def get_recommended_spots(user_input):
    chatbot = ChatCompletion([user_input])
    results = chatbot.complete()
    return [x for x in results[0] if x in spot_list and x in locations_df["名称"].tolist()][:8]  # 限制最多8个景点

# 用户输入和按钮
user_input = st.text_area("请输入您的游览需求：", value="我喜欢去在校学生多的地方")
if st.button("获取推荐景点并规划路径"):
    # 调用大模型获取推荐景点
    with st.spinner("正在获取推荐景点..."):
        recommended_spots = get_recommended_spots(user_input)
        if not recommended_spots:
            st.error("❌ 无法获取有效推荐景点，请检查输入或 API 配置。")
        else:
            st.success(f"✅ 推荐的景点序列：{recommended_spots}")

    # 使用FeatureGroup分层管理标记（恢复默认绿色标记）
    marker_group = folium.FeatureGroup(name="Attractions").add_to(m)
    marker_dict = {}
    for _, row in locations_df[locations_df["名称"].isin(recommended_spots)].iterrows():
        folium.Marker(
            location=[row["纬度"], row["经度"]],
            tooltip=row["名称"],
            popup=row["名称"],
            # 移除自定义图标，使用默认绿色标记
        ).add_to(marker_group)
        marker_dict[row["名称"]] = (row["纬度"], row["经度"])

    # 路径规划
    selected_nodes = recommended_spots
    if len(selected_nodes) < 2:
        st.warning("请确保推荐的景点数量至少为2个")
    else:
        # 路径过滤（进一步限制边数）
        Filtered_df = df[df["起点"].isin(selected_nodes) & df["终点"].isin(selected_nodes)]
        filtered_df = Filtered_df[['起点', '终点', '预计步行时间_分钟']]
        simple_Filtered_df = simple_df[simple_df["起点"].isin(selected_nodes) & simple_df["终点"].isin(selected_nodes)]
        simple_filtered_df = simple_Filtered_df[['起点', '终点', '预计步行时间_分钟']]
        
        # 限制子图规模（最多保留 50 条边）
        if len(filtered_df) > 50:
            filtered_df = filtered_df.nsmallest(50, '预计步行时间_分钟')
        if len(simple_filtered_df) > 50:
            simple_filtered_df = simple_filtered_df.nsmallest(50, '预计步行时间_分钟')
        
        # 创建边列表
        edges = list(filtered_df.itertuples(index=False, name=None))
        edge_pairs = [(u, v) for u, v, _ in edges]
        simple_edges = list(simple_filtered_df.itertuples(index=False, name=None))
        simple_edge_pairs = [(u, v) for u, v, _ in simple_edges]

        # 初始化优化类
        ecp = EulerProgramming()
        ecp.addEdges(edge_pairs)
        hmp = HamiltonProgramming()
        hmp.addEdges(simple_edge_pairs)
        best_path = None
        best_time = float("inf")
        best_type = None

        # 判断哈密顿通路
        with st.spinner("Gurobi 正在判断哈密顿通路..."):
            try:
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
            except Exception as e:
                st.error(f"❌ 哈密顿路径求解失败: {str(e)}")

        # 判断欧拉路径
        with st.spinner("Gurobi 正在判断欧拉路径..."):
            try:
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
            except Exception as e:
                st.error(f"❌ 欧拉路径求解失败: {str(e)}")

        # 保存结果到 session_state
        if best_path is not None:
            st.session_state.best_path = best_path
            st.session_state.best_time = best_time
            st.session_state.best_type = best_type
        else:
            st.error("❌ 无可行路径")

        # 显示结果
        if "best_path" in st.session_state:
            best_path = st.session_state.best_path
            best_time = st.session_state.best_time
            best_type = st.session_state.best_type

            st.markdown(f"🏆 **更优方案为：{best_type}**")
            st.markdown(f"🔀 路径：{' -> '.join([str(u) for u, v in best_path] + [str(best_path[-1][1])])}")
            st.markdown(f"🕒 预计最短步行时间：{best_time:.1f} 分钟")

            # 优化路径坐标生成：采样关键点
            df["路径节点"] = df["路径节点"].apply(ast.literal_eval)
            
            def get_full_path_coordinates(best_path, path_detail_df, node_id_to_coord, max_points=100):
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
                # 采样路径点，限制最大点数
                if len(full_node_list) > max_points:
                    step = len(full_node_list) // max_points
                    full_node_list = full_node_list[::step][:max_points]
                return [node_id_to_coord[node] for node in full_node_list if node in node_id_to_coord]
            
            # 获取节点坐标
            node_id_to_coord = dict(zip(node_coordinates_df["node"], zip(node_coordinates_df["lat"], node_coordinates_df["lng"])))
            
            # 绘制路径（使用FeatureGroup，保持红色）
            path_group = folium.FeatureGroup(name="Path").add_to(m)
            path_coords = get_full_path_coordinates(best_path, df, node_id_to_coord)
            if path_coords:
                folium.PolyLine(
                    path_coords,
                    color="red",
                    weight=5,
                    opacity=0.8,
                    tooltip=f"{best_type}，预计 {best_time:.1f} 分钟"
                ).add_to(path_group)

    # 添加图层控制
    folium.LayerControl().add_to(m)

# 显示地图（优化尺寸）
st_data = st_folium(m, width=600, height=400, returned_objects=[])
