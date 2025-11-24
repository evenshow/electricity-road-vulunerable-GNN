"""
通用工具函数模块，负责图数据的预处理与电力网络的环境初始化。
"""

import yaml
import json
import random
import os
import networkx as nx

from model import ElecNoStep

def str2int(json_data):
    """将 JSON 中的键从字符串转换为整数，便于后续运算。"""

    new_dict = {}
    for key, value in json_data.items():
        new_dict[int(key)] = value
    return new_dict

def init_env():
    """读取电力网络的配置和拓扑，返回电力环境对象。"""

    with open("./data/electricity/config.yml") as f:
            config = yaml.safe_load(f)
       
    with open('./data/electricity/power_10kv.json') as json_file:
            power_10 = json.load(json_file)

    with open('./data/electricity/power_110kv.json') as json_file:
        power_load = json.load(json_file)
        
    power_10 = str2int(power_10)
    power_load = str2int(power_load)
    
    with open('./data/electricity/all_dict_correct.json') as json_file:
        topology = json.load(json_file)
    
    for key in topology:
        topology[key] = str2int(topology[key])
    elec = ElecNoStep(config, topology, power_10, power_load)

    return elec

def calculate_size_of_gcc(Graph):
    """计算图中最大连通子图的节点数量。"""

    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)]
    size_of_gcc = max(size_of_connected_components)

    return size_of_gcc

def calculate_pairwise_connectivity(Graph):
    """计算节点对之间的可达性，用于衡量连通性。"""

    size_of_connected_components = [len(part_graph) for part_graph in nx.connected_components(Graph)]
    element_of_pc  = [size*(size - 1)/2 for size in size_of_connected_components]
    pairwise_connectivity = sum(element_of_pc)

    return pairwise_connectivity

def influenced_tl_by_elec(elec_state, elec2road, tgraph):
    """根据电力状态映射出受影响的交通灯节点。"""
    elec10kv = []
    # 汇总三类受损节点
    for key in ['ruined', 'cascaded', 'stopped']:
        elec10kv += elec_state[5][key]
    elec10kv = [str(node) for node in elec10kv if str(node) in elec2road.keys()]
    tl_id = [elec2road[node] for node in elec10kv if elec2road[node] in tgraph.nodes()]
    return tl_id

def nodes_ranked_by_CI(Graph):
    """
    计算网络中各点的组合影响(CI),由大到小将点排序
    return->list
    """
    node_degree_map = dict(nx.degree(Graph))
    node_ci = []
    for node in node_degree_map:
        ci = 0
        neighbors = list(Graph.neighbors(node))
        for neighbor in neighbors:
            ci += (node_degree_map[neighbor]-1)
        node_ci.append((node,ci*(node_degree_map[node]-1)))
    list.sort(node_ci,key = lambda y:y[1],reverse=True)
    nodes = [example[0] for example in node_ci]

    return nodes
    
def nodes_ranked_by_Degree(Graph):
    """
    计算网络中各点的度，由大到小将点排序
    return->list
    """
    node_degree = list(nx.degree(Graph))
    list.sort(node_degree,key = lambda y:y[1],reverse=True)
    nodes = [example[0] for example in node_degree]

    return nodes
