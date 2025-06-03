import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
import folium
from streamlit_folium import st_folium
import ast
import sys
from time import sleep
import os
from tqdm import tqdm
import re
import json
import concurrent.futures
from openai import OpenAI
from utils import handle_path_planning

# 增加递归深度限制
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
max_points = 200

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
        json_str = re.sub(r'[\n\r]', ' ', json_str)
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
    df = pd.read_csv("./data/pku_all_simple_paths.csv")
    locations_df = pd.read_csv("./data/pku_locations_updated.csv")
    node_coordinates_df = pd.read_csv("./data/pku_walk_node_locations.csv")
    
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
st.markdown("您可以通过大语言模型推荐景点，或手动选择景点，然后查看哈密顿通路和欧拉通路的对比。")

# 初始化地图中心
center = [39.992, 116.305]  # 未名湖附近

# 缓存 API 调用结果
@st.cache_data
def get_recommended_spots(user_input):
    chatbot = ChatCompletion([user_input])
    results = chatbot.complete()
    return [x for x in results[0] if x in spot_list and x in locations_df["名称"].tolist()][:8]

# 用户输入区域
st.subheader("选择您的游览方式")
llm_tab, manual_tab,security_tab = st.tabs(["大语言模型推荐", "手动选择景点","安保巡逻"])

# 存储推荐或选择的景点及路径结果
if 'selected_spots' not in st.session_state:
    st.session_state.selected_spots = []
if 'path_results' not in st.session_state:
    st.session_state.path_results = {}

with llm_tab:
    user_input = st.text_area("请输入您的游览需求：", value="我喜欢去热闹的地方", key="llm_input")

    if st.button("通过大语言模型推荐"):
        with st.spinner("正在获取推荐景点..."):
            recommended_spots = get_recommended_spots(user_input)
            st.write(f"[推荐景点]:{recommended_spots}")
            if not recommended_spots:
                st.error("❌ 无法获取有效推荐景点，请检查输入或 API 配置。")
            else:
                st.session_state.selected_spots = recommended_spots
                st.success(f"✅ 推荐的景点序列：{recommended_spots}")

    if "selected_spots" in st.session_state and st.session_state.selected_spots:
        st.markdown("### 使用推荐结果进行路径规划")
        is_shortest_only = st.checkbox("仅进行最短路规划", key="llm_shortest_only")

        if st.button("开始规划路径", key="llm_plan_button"):
            with st.spinner("正在规划路径..."):
                handle_path_planning(
                    st.session_state.selected_spots,
                    df,
                    simple_df,
                    locations_df, 
                    node_coordinates_df,
                    center, 
                    boundary_coords, 
                    max_points,
                    is_shortest_only=is_shortest_only,
                )

with manual_tab:
    selected_spots = st.multiselect("请选择您想游览的景点（至少2个，最多8个）：", options=spot_list, default=None, key="manual_select")
    is_shortest_only = st.checkbox("仅进行最短路规划", key="manual_shortest_only")
    if st.button("开始规划路径", key="manual_plan_button"):
        if len(selected_spots) < 2:
            st.error("❌ 请至少选择2个景点！")
        elif len(selected_spots) > 8:
            st.error("❌ 最多选择8个景点！")
        else:
            st.success(f"✅ 已选择景点：{selected_spots}")
            with st.spinner("正在规划路径..."):
                handle_path_planning(selected_spots, df, simple_df, locations_df, node_coordinates_df,center, boundary_coords, max_points,is_shortest_only)


with security_tab:
    selected_spots = st.multiselect("请选择您想巡逻的位置（至少2个，最多8个）：", options=spot_list, key="security_select")
    if st.button("开始规划路径", key="security_plan_button"):
        if len(selected_spots) < 2:
            st.error("❌ 请至少选择2个景点！")
        elif len(selected_spots) > 8:
            st.error("❌ 最多选择8个景点！")
        else:
            st.success(f"✅ 已选择景点：{selected_spots}")
            with st.spinner("正在规划路径..."):
                handle_path_planning(selected_spots, 
                                     df, 
                                     simple_df, 
                                     locations_df, 
                                     node_coordinates_df,
                                     center, 
                                     boundary_coords, 
                                     max_points,
                                     is_shortest_only=None)
