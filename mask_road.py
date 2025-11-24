"""
对道路图进行节点移除与特征扰动实验的脚本，增加中文注释便于理解流程。
"""

from model import TraGraph, GCN, construct_negative_graph, compute_loss
from utils import init_env, calculate_pairwise_connectivity

import time
import torch
import torch.nn as nn
import dgl
import os
import random
import numpy as np
import networkx as nx
import argparse

# 命令行参数解析器，控制训练超参数
parser = argparse.ArgumentParser(description='elec attack')

parser.add_argument('--epoch', type=int, default=1000, help='Times to train')
parser.add_argument('--batch', type=int, default=20, help='Data number used to train')
parser.add_argument('--gamma', type=float, default=0.9, help='Related to further reward')
parser.add_argument('--lr', type=float, default=0.01, help='Laerning rate')
parser.add_argument('--epsilon', type=float, default=0.6, help='Epsilon greedy policy')

args = parser.parse_args()

# 嵌入与扰动文件路径
# tpt = './embedding/primary.pt'
tpt = './embedding/tra_feat.pt'
perturb_pth = './embedding/perturb_primary.pt'
# perturb_pth = './embedding/perturb_ter.pt'

# 读取道路数据的文件路径
TFILE1 = './data/road/road_junc_map.json'
TFILE2 = './data/road/road_type_map.json'
TFILE3 = './data/road/tl_id_road2elec_map.json'

# 图神经网络超参数
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP = 5
EPOCH = args.epoch
LR = args.lr
BATCH_SIZE = args.batch
GAMMA = args.gamma
EPSILON = args.epsilon
MEMORY_CAPACITY = 1000
TARGET_REPLACE_ITER = 25

random.seed(20230124)
device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # 构建交通图并加载节点特征
    tgraph = TraGraph(
        file1=TFILE1,
        file2=TFILE2,
        file3=TFILE3,
        embed_dim=EMBED_DIM,
        hid_dim=HID_DIM,
        feat_dim=FEAT_DIM,
        r_type='primary',
        khop=KHOP,
        epochs=300,
        pt_path=tpt,
    )

    node_num = tgraph.node_num

    tnodes = list(tgraph.nxgraph.nodes())
    tedges = list(tgraph.nxgraph.edges())

    # 打乱边并移除一小部分作为缺失场景
    random.shuffle(tedges)
    tedges = tedges[:-10]

    perturb_graph = nx.Graph()
    perturb_graph.add_nodes_from(tnodes)
    perturb_graph.add_edges_from(tedges)

    # 计算度排序攻击的连通性下降
    degree = dict(nx.degree(perturb_graph))
    degree_list = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:20]
    tgc = perturb_graph.copy()
    orig_val = calculate_pairwise_connectivity(tgc)
    result = []
    for id, (node, degree) in enumerate(degree_list):
        tgc.remove_node(node)
        val = calculate_pairwise_connectivity(tgc) / orig_val
        result.append([id + 1, val])

    result = np.array(result)
    np.savetxt('./results/mask_road_degree.txt', result)

    # 计算组合影响（CI）排序攻击的连通性下降
    CI = []
    degree = dict(nx.degree(perturb_graph))
    for node in degree:
        ci = 0
        neighbors = list(perturb_graph.neighbors(node))
        for neighbor in neighbors:
            ci += (degree[neighbor] - 1)
        CI.append((node, ci * (degree[node] - 1)))

    result = []
    ci_list = sorted(CI, key=lambda x: x[1], reverse=True)[:20]
    tgc = tgraph.nxgraph.copy()
    for id, (node, CI) in enumerate(ci_list):
        tgc.remove_node(node)
        val = calculate_pairwise_connectivity(tgc) / orig_val
        result.append([id + 1, val])

    result = np.array(result)
    np.savetxt('./results/mask_road_ci.txt', result)

    # 将扰动图转换为 DGL 图并复制特征
    perturb_graph = dgl.from_networkx(perturb_graph)
    orig_feat = tgraph.feat.detach()

    perturb_graph.ndata['feat'] = orig_feat.clone().requires_grad_()

    gcn = GCN(EMBED_DIM, HID_DIM, FEAT_DIM)
    optimizer = torch.optim.Adam(gcn.parameters())
    optimizer.zero_grad()

    # 通过对比损失逼近原始特征，同时保持结构扰动
    for epoch in range(500):
        t = time.time()
        negative_graph = construct_negative_graph(perturb_graph, 5)
        pos_score, neg_score = gcn(perturb_graph, negative_graph, perturb_graph.ndata['feat'])
        feat = gcn.sage(perturb_graph, perturb_graph.ndata['feat'])
        dist = torch.dist(feat, orig_feat, p=2)

        loss = compute_loss(pos_score, neg_score) + dist * 10
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(
                "Epoch:",
                '%03d' % (epoch + 1),
                " train_loss = ",
                "{:.5f} ".format(loss.item()),
                " time=",
                "{:.4f}s".format(time.time() - t)
            )

    print(" train_loss = ", "{:.5f} ".format(loss.item()))

    # 保存扰动后的节点特征
    feat = gcn.sage(perturb_graph, perturb_graph.ndata['feat'])
    torch.save(feat, perturb_pth)
