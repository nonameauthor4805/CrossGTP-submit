import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import os
import dgl
import argparse
from dgl.nn import GATConv
from sklearn.feature_extraction.text import TfidfTransformer
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--scity', type=str, default='NY')
parser.add_argument("--tcity", type=str, default='DC')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--emb_dim', type=int, default=64)
parser.add_argument('--topk', type=int, default=15)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mmd_weight', type=float, default=10)
parser.add_argument('--et_weight', type=float, default=10)
args = parser.parse_args()
num_gat_layers = args.num_layers
learning_rate = args.learning_rate
weight_decay = args.weight_decay
num_heads = args.num_heads
hidden_dim = args.hidden_dim
emb_dim = args.emb_dim
topk = args.topk
num_epochs = args.num_epochs
gpu = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

scity = args.scity 
tcity = args.tcity
start_time = time.time() 
dataname = {
    'NY': 'Taxi', 
    'DC': 'Taxi', 
    'CHI': 'Taxi', 
    'Porto': 'Taxi', 
    'BOS': 'Bike'
}
source_dataname = dataname[scity]
target_dataname = dataname[tcity]
if tcity == 'BOS':
    source_dataname = 'Bike'
save_path = '../embeddings/%s_%s/' % (scity, tcity)
if not os.path.exists(save_path):
    os.makedirs(save_path)
if os.path.exists(save_path + 'best_result'):
    with open(save_path + 'best_result', 'r') as infile:
        lines = []
        for line in infile:
            lines.append(line.rstrip())
        best_result = eval(lines[0])
        previous_best = eval(lines[0])
else:
    best_result = 0
print("Current best result %.4f" % best_result)    
    


source_poi_feat = np.load("../data/%s/%s_poi.npy" % (scity, scity))
target_poi_feat = np.load("../data/%s/%s_poi.npy" % (tcity, tcity))
lng_s = source_poi_feat.shape[0]
lat_s = source_poi_feat.shape[1]
lng_t = target_poi_feat.shape[0]
lat_t = target_poi_feat.shape[1]
source_poi_feat = source_poi_feat.reshape(lng_s * lat_s, -1)
target_poi_feat = target_poi_feat.reshape(lng_t * lat_t, -1)
source_poi_feat_sum = source_poi_feat.sum(1)
target_poi_feat_sum = target_poi_feat.sum(1)
source_poi_label = masked_percentile_label(source_poi_feat_sum)
target_poi_label = masked_percentile_label(target_poi_feat_sum)
transform = TfidfTransformer()
source_norm_poi = np.array(transform.fit_transform(source_poi_feat).todense())
transform = TfidfTransformer()
target_norm_poi = np.array(transform.fit_transform(target_poi_feat).todense())
in_dim = source_norm_poi.shape[1]

def load_transport_data(cityname):
    ret_dict = {}
    if cityname != 'BOS':
        taxi_pickup = np.load("../data/%s/Taxi%s_pickup.npy" % (cityname, cityname))
        taxi_pickup_sum = taxi_pickup.sum(0).reshape(-1)
        taxi_pickup_mask = taxi_pickup_sum > 0
        taxi_pickup_label = masked_percentile_label(taxi_pickup_sum, taxi_pickup_mask)

        taxi_dropoff = np.load("../data/%s/Taxi%s_dropoff.npy" % (cityname, cityname))
        taxi_dropoff_sum = taxi_dropoff.sum(0).reshape(-1)
        taxi_dropoff_mask = taxi_dropoff_sum > 0
        taxi_dropoff_label = masked_percentile_label(taxi_dropoff_sum, taxi_dropoff_mask)
        ret_dict['taxi_pickup'] = (taxi_pickup_mask, taxi_pickup_label)
        ret_dict['taxi_dropoff'] = (taxi_dropoff_mask, taxi_dropoff_label)
    if cityname != 'Porto':
        bike_pickup = np.load("../data/%s/Bike%s_pickup.npy" % (cityname, cityname))
        bike_pickup_sum = bike_pickup.sum(0).reshape(-1)
        bike_pickup_mask = bike_pickup_sum > 0
        bike_pickup_label = masked_percentile_label(bike_pickup_sum, bike_pickup_mask)

        bike_dropoff = np.load("../data/%s/Bike%s_dropoff.npy" % (cityname, cityname))
        bike_dropoff_sum = bike_dropoff.sum(0).reshape(-1)
        bike_dropoff_mask = bike_dropoff_sum > 0
        bike_dropoff_label = masked_percentile_label(bike_dropoff_sum, bike_dropoff_mask)
        ret_dict['bike_pickup'] = (bike_pickup_mask, bike_pickup_label)
        ret_dict['bike_dropoff'] = (bike_dropoff_mask, bike_dropoff_label)        
    return ret_dict

source_transport = load_transport_data(scity)
target_transport = load_transport_data(tcity)

source_prox_adj = add_self_loop(build_prox_graph(lng_s, lat_s))
target_prox_adj = add_self_loop(build_prox_graph(lng_t, lat_t))
source_road_adj = add_self_loop(build_road_graph(scity, lng_s, lat_s))
target_road_adj = add_self_loop(build_road_graph(tcity, lng_t, lat_t))
source_poi_adj, source_poi_cos = build_poi_graph(source_norm_poi, topk)
target_poi_adj, target_poi_cos = build_poi_graph(target_norm_poi, topk)
source_poi_adj = add_self_loop(source_poi_adj)
target_poi_adj = add_self_loop(target_poi_adj)
source_s_adj, source_d_adj, source_od_adj = build_source_dest_graph(scity, source_dataname, lng_s, lat_s, topk)
target_s_adj, target_d_adj, target_od_adj = build_source_dest_graph(tcity, target_dataname, lng_t, lat_t, topk)
print("Source graphs: ")
print("prox_adj: %d nodes, %d edges" % (source_prox_adj.shape[0], np.sum(source_prox_adj)))
print("road adj: %d nodes, %d edges" % (source_road_adj.shape[0], np.sum(source_road_adj > 0)))
print("poi_adj, %d nodes, %d edges" % (source_poi_adj.shape[0], np.sum(source_poi_adj > 0)))
print("s_adj, %d nodes, %d edges" % (source_s_adj.shape[0], np.sum(source_s_adj > 0)))
print("d_adj, %d nodes, %d edges" % (source_d_adj.shape[0], np.sum(source_d_adj > 0)))
print()
print("Target graphs:")
print("prox_adj: %d nodes, %d edges" % (target_prox_adj.shape[0], np.sum(target_prox_adj)))
print("road adj: %d nodes, %d edges" % (target_road_adj.shape[0], np.sum(target_road_adj > 0)))
print("poi_adj, %d nodes, %d edges" % (target_poi_adj.shape[0], np.sum(target_poi_adj > 0)))
print("s_adj, %d nodes, %d edges" % (target_s_adj.shape[0], np.sum(target_s_adj > 0)))
print("d_adj, %d nodes, %d edges" % (target_d_adj.shape[0], np.sum(target_d_adj > 0)))
print()

source_graphs = adjs_to_graphs([source_prox_adj, source_road_adj, source_poi_adj, source_s_adj, source_d_adj])
target_graphs = adjs_to_graphs([target_prox_adj, target_road_adj, target_poi_adj, target_s_adj, target_d_adj])
for i in range(len(source_graphs)):
    source_graphs[i] = source_graphs[i].to(device)
    target_graphs[i] = target_graphs[i].to(device)

# define model
class MVGAT(nn.Module):
    def __init__(self, num_graphs=3, num_gat_layer=2, in_dim=14, hidden_dim=64, emb_dim = 64, num_heads=2, residual=True):
        super().__init__()
        self.num_graphs = num_graphs
        self.num_gat_layer = num_gat_layer
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.residual = residual

        # multiple GATs
        self.multi_gats = nn.ModuleList()
        for j in range(self.num_gat_layer):
            gats = nn.ModuleList()
            for i in range(self.num_graphs):
                if j == 0:
                    gats.append(GATConv(self.in_dim, 
                                        self.hidden_dim, 
                                        self.num_heads, 
                                        residual = self.residual, 
                                        allow_zero_in_degree = True))
                elif j == self.num_gat_layer - 1:
                    gats.append(GATConv(self.hidden_dim * self.num_heads, 
                                        self.emb_dim // self.num_heads, 
                                        self.num_heads, 
                                        residual = self.residual, 
                                        allow_zero_in_degree = True))
                else:
                    gats.append(GATConv(self.hidden_dim * self.num_heads, 
                                        self.hidden_dim, 
                                        self.num_heads, 
                                        residual = self.residual, 
                                        allow_zero_in_degree = True))
            self.multi_gats.append(gats)
        
        # attention fusion modules
        # self.fusion_linear = nn.Linear(self.emb_dim, self.emb_dim)

        # self.self_q = nn.ModuleList()
        # self.self_k = nn.ModuleList()
        # for i in range(self.num_graphs):
        #     self.self_q.append(nn.Linear(self.emb_dim, self.emb_dim))
        #     self.self_k.append(nn.Linear(self.emb_dim, self.emb_dim))
        
        # self.edge_network = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim), 
        #                                   nn.ReLU(), 
        #                                   nn.Linear(self.emb_dim, self.num_graphs))

    def forward(self, graphs, feat):
        views = []
        for i in range(self.num_graphs):
            for j in range(self.num_gat_layer):
                if j == 0:
                    z = self.multi_gats[j][i](graphs[i], feat)
                else:
                    # z = self.multi_gats[j][i](graphs[i], next_in[i])
                    z = self.multi_gats[j][i](graphs[i], z)
                z = F.relu(z).flatten(1)
            views.append(z)
        return views

class FusionModule(nn.Module):
    def __init__(self, num_graphs, emb_dim, alpha):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.alpha = alpha

        self.fusion_linear = nn.Linear(self.emb_dim, self.emb_dim)
        self.self_q = nn.ModuleList()
        self.self_k = nn.ModuleList()
        for i in range(self.num_graphs):
            self.self_q.append(nn.Linear(self.emb_dim, self.emb_dim))
            self.self_k.append(nn.Linear(self.emb_dim, self.emb_dim))

    def forward(self, views):        
        # run fusion by self attention
        cat_views = torch.stack(views, dim = 0)
        self_attentions = []
        for i in range(self.num_graphs):
            Q = self.self_q[i](cat_views)
            K = self.self_k[i](cat_views)
            # (3, num_nodes, 64)
            attn = F.softmax(torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.emb_dim), dim = -1)
            # (3, num_nodes, num_nodes)
            output = torch.matmul(attn, cat_views)
            self_attentions.append(output)
        self_attentions = sum(self_attentions) / self.num_graphs
        # (3, num_nodes, 64 * 2)
        for i in range(self.num_graphs):
            views[i] = self.alpha * self_attentions[i] + (1-self.alpha) * views[i]

        # further run multi-view fusion
        mv_outputs = []
        for i in range(self.num_graphs):
            mv_outputs.append(torch.sigmoid(self.fusion_linear(views[i])) * views[i])
        
        fused_outputs = sum(mv_outputs)
        # next_in = [(view + fused_outputs) / 2 for view in views]
        return fused_outputs, [(views[i] + fused_outputs) / 2 for i in range(self.num_graphs)]

class EdgeTypeDiscriminator(nn.Module):
    def __init__(self, num_graphs, emb_dim):
        super().__init__()
        self.num_graphs = num_graphs
        self.emb_dim = emb_dim
        self.edge_network = nn.Sequential(nn.Linear(2 * self.emb_dim, self.emb_dim), 
                                          nn.ReLU(), 
                                          nn.Linear(self.emb_dim, self.num_graphs))
    def forward(self, src_embs, dst_embs):
        edge_vec = torch.cat([src_embs, dst_embs], dim = 1)
        return self.edge_network(edge_vec)

def graphs_to_edge_labels(graphs):
    edge_label_dict = {}
    for i, graph in enumerate(graphs):
        src, dst = graph.edges()
        for s, d in zip(src, dst):
            s = s.item()
            d = d.item()
            if (s, d) not in edge_label_dict:
                edge_label_dict[(s, d)] = np.zeros(len(graphs))
            edge_label_dict[(s, d)][i] = 1
    edges = []
    edge_labels = [] 
    for k in edge_label_dict.keys():
        edges.append(k)
        edge_labels.append(edge_label_dict[k])
    edges = np.array(edges)
    edge_labels = np.array(edge_labels)
    return edges, edge_labels

def evaluate(emb, label, mask = None):
    if mask is not None:
        emb = emb[mask, :]
    # evaluate on downstream tasks
    # prepare 5 tasks: POI, taxi pickup, taxi dropoff, bike pickup, bike dropoff
    logreg = LogisticRegression(max_iter = 500)
    results = cross_validate(logreg, emb, label)
    return results['test_score'].mean()

source_edges, source_edge_labels = graphs_to_edge_labels(source_graphs)
target_edges, target_edge_labels = graphs_to_edge_labels(target_graphs)
print("Source total number of edges: ", len(source_edges))
print("Target total number of edges: ", len(target_edges))

mvgat = MVGAT(len(source_graphs), num_gat_layers, in_dim, hidden_dim, emb_dim, num_heads).to(device)
source_fusion = FusionModule(len(source_graphs), emb_dim, 0.8).to(device)
target_fusion = FusionModule(len(target_graphs), emb_dim, 0.8).to(device)
edge_type_disc = EdgeTypeDiscriminator(len(source_graphs), emb_dim).to(device)
param_list = list(mvgat.parameters()) + list(source_fusion.parameters()) + list(target_fusion.parameters()) + list(edge_type_disc.parameters())
optimizer = optim.Adam(param_list, lr = learning_rate, weight_decay = weight_decay)

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
mmd = MMD_loss()

for ep in range(num_epochs):
    # compute on source
    source_views = mvgat(source_graphs, torch.Tensor(source_norm_poi).to(device))
    source_fused_emb, source_embs = source_fusion(source_views)
    # prox, road, poi, s, d
    source_s_emb = source_embs[-2]
    source_d_emb = source_embs[-1]
    source_poi_emb = source_embs[-3]
    source_recons_sd = torch.matmul(source_s_emb, source_d_emb.transpose(0, 1))
    source_pred_d = torch.log(torch.softmax(source_recons_sd, dim = 1) + 1e-5)
    source_loss_d = (torch.Tensor(source_od_adj).to(device) * source_pred_d).mean()

    source_pred_s = torch.log(torch.softmax(source_recons_sd.transpose(0, 1), dim = 1) + 1e-5)
    source_loss_s = (torch.Tensor(source_od_adj).to(device) * source_pred_s).mean()

    source_poi_sim = torch.matmul(source_poi_emb, source_poi_emb.transpose(0, 1))  
    source_loss_poi = ((source_poi_sim - torch.Tensor(source_poi_cos).to(device)) ** 2).mean()
    source_loss = -source_loss_d - source_loss_s + source_loss_poi
    
    # compute on target
    target_views = mvgat(target_graphs, torch.Tensor(target_norm_poi).to(device))
    target_fused_emb, target_embs = target_fusion(target_views)
    # prox, road, poi, s, d
    target_s_emb = target_embs[-2]
    target_d_emb = target_embs[-1]
    target_poi_emb = target_embs[-3]
    target_recons_sd = torch.matmul(target_s_emb, target_d_emb.transpose(0, 1))
    target_pred_d = torch.log(torch.softmax(target_recons_sd, dim = 1) + 1e-5)
    target_loss_d = (torch.Tensor(target_od_adj).to(device) * target_pred_d).mean()

    target_pred_s = torch.log(torch.softmax(target_recons_sd.transpose(0, 1), dim = 1) + 1e-5)
    target_loss_s = (torch.Tensor(target_od_adj).to(device) * target_pred_s).mean()

    target_poi_sim = torch.matmul(target_poi_emb, target_poi_emb.transpose(0, 1))  
    target_loss_poi = ((target_poi_sim - torch.Tensor(target_poi_cos).to(device)) ** 2).mean()
    target_loss = -target_loss_d - target_loss_s + target_loss_poi

    # compute mmd loss
    source_ids = np.random.randint(0, lng_s * lat_s, size = (256,))
    target_ids = np.random.randint(0, lng_t * lat_t, size = (256,))
    mmd_loss = mmd(source_fused_emb[source_ids, :], target_fused_emb[target_ids, :])
    mmd_loss += sum([mmd(source_embs[i][source_ids, :], target_embs[i][target_ids, :]) for i in range(len(source_graphs))])
    
    # train edge type
    source_batch_edges = np.random.randint(0, len(source_edges), size = (1024, ))
    target_batch_edges = np.random.randint(0, len(target_edges), size = (1024, ))
    source_batch_src = torch.Tensor(source_edges[source_batch_edges, 0]).long()
    source_batch_dst = torch.Tensor(source_edges[source_batch_edges, 1]).long()
    source_emb_src = source_fused_emb[source_batch_src, :]
    source_emb_dst = source_fused_emb[source_batch_dst, :]
    
    target_batch_src = torch.Tensor(target_edges[target_batch_edges, 0]).long()
    target_batch_dst = torch.Tensor(target_edges[target_batch_edges, 1]).long()
    target_emb_src = target_fused_emb[target_batch_src, :]
    target_emb_dst = target_fused_emb[target_batch_dst, :]
    pred_source = edge_type_disc.forward(source_emb_src, source_emb_dst)
    pred_target = edge_type_disc.forward(target_emb_src, target_emb_dst)
    source_batch_labels = torch.Tensor(source_edge_labels[source_batch_edges]).to(device)
    target_batch_labels = torch.Tensor(target_edge_labels[target_batch_edges]).to(device)
    loss_et_source = -((source_batch_labels * torch.log(torch.sigmoid(pred_source) + 1e-6)) + (1 - source_batch_labels) * torch.log(1 - torch.sigmoid(pred_source) + 1e-6)).sum(1).mean()
    loss_et_target = -((target_batch_labels * torch.log(torch.sigmoid(pred_target) + 1e-6)) + (1 - target_batch_labels) * torch.log(1 - torch.sigmoid(pred_target) + 1e-6)).sum(1).mean()
    loss_et = loss_et_source + loss_et_target
    if tcity == 'Porto':
        if scity == 'NY':
            source_loss = source_loss / 4
        else:
            source_loss = source_loss / 2
    loss = source_loss + target_loss + mmd_loss * args.mmd_weight + loss_et * args.et_weight
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("[%.2fs]Epoch %d, source loss d %.5f, loss s %.5f, loss poi %.5f" % (time.time() - start_time, ep, source_loss_d.item(), source_loss_s.item(), source_loss_poi.item()))
    print("[%.2fs]Epoch %d, target loss d %.5f, loss s %.5f, loss poi %.5f" % (time.time() - start_time, ep, target_loss_d.item(), target_loss_s.item(), target_loss_poi.item()))
    print("[%.2fs]Epoch %d, mmd loss %.5f, et loss %.5f" % (time.time() - start_time, ep, mmd_loss.item(), loss_et.item()))

    # evaluate source
    source_emb_arr = source_fused_emb.detach().cpu().numpy()
    print("Source embedding arr", source_emb_arr.shape)
    if scity != 'BOS':
        # evaluate taxi pickup
        cvscore_pickup = evaluate(source_emb_arr, source_transport['taxi_pickup'][1], source_transport['taxi_pickup'][0])
        # evaluate taxi dropoff
        cvscore_dropoff = evaluate(source_emb_arr, source_transport['taxi_dropoff'][1], source_transport['taxi_dropoff'][0])
        print("[%.2fs]Epoch %d, source cross_validate score on taxi pickup and dropoff %.4f and %.4f" % (time.time() - start_time, ep, cvscore_pickup, cvscore_dropoff))
    if scity != 'Porto':
        # evaluate taxi pickup
        cvscore_pickup = evaluate(source_emb_arr, source_transport['bike_pickup'][1], source_transport['bike_pickup'][0])
        # evaluate taxi dropoff
        cvscore_dropoff = evaluate(source_emb_arr, source_transport['bike_dropoff'][1], source_transport['bike_dropoff'][0])
        print("[%.2fs]Epoch %d, source cross_validate score on bike pickup and dropoff %.4f and %.4f" % (time.time() - start_time, ep, cvscore_pickup, cvscore_dropoff))

    # evaluate target
    target_emb_arr = target_fused_emb.detach().cpu().numpy()
    print("Target embedding arr", target_emb_arr.shape)
    if tcity != 'BOS':
        # evaluate taxi pickup
        cvscore_pickup = evaluate(target_emb_arr, target_transport['taxi_pickup'][1], target_transport['taxi_pickup'][0])
        # evaluate taxi dropoff
        cvscore_dropoff = evaluate(target_emb_arr, target_transport['taxi_dropoff'][1], target_transport['taxi_dropoff'][0])
        print("[%.2fs]Epoch %d, target cross_validate score on taxi pickup and dropoff %.4f and %.4f" % (time.time() - start_time, ep, cvscore_pickup, cvscore_dropoff))
    if tcity != 'Porto':
        # evaluate taxi pickup
        cvscore_pickup = evaluate(target_emb_arr, target_transport['bike_pickup'][1], target_transport['bike_pickup'][0])
        # evaluate taxi dropoff
        cvscore_dropoff = evaluate(target_emb_arr, target_transport['bike_dropoff'][1], target_transport['bike_dropoff'][0])
        print("[%.2fs]Epoch %d, target cross_validate score on bike pickup and dropoff %.4f and %.4f" % (time.time() - start_time, ep, cvscore_pickup, cvscore_dropoff))

    # domain adaptive evaluation
    da_result = 0
    logreg = LogisticRegression(max_iter=500)
    print("Domain adaptive evaluation")
    if tcity != 'BOS':
        
        source_taxi_pickup_mask = source_transport['taxi_pickup'][0]
        source_taxi_pickup_label = source_transport['taxi_pickup'][1]
        source_taxi_dropoff_mask = source_transport['taxi_dropoff'][0]
        source_taxi_dropoff_label = source_transport['taxi_dropoff'][1]
        target_taxi_pickup_mask = target_transport['taxi_pickup'][0]
        target_taxi_pickup_label = target_transport['taxi_pickup'][1]
        target_taxi_dropoff_mask = target_transport['taxi_dropoff'][0]
        target_taxi_dropoff_label = target_transport['taxi_dropoff'][1]

        logreg = LogisticRegression(max_iter=500)
        logreg.fit(source_emb_arr[source_taxi_pickup_mask], source_taxi_pickup_label)
        target_taxi_pickup_pred = logreg.predict(target_emb_arr[target_taxi_pickup_mask])
        target_taxi_pickup_acc = (target_taxi_pickup_pred == target_taxi_pickup_label).mean()
        
        logreg = LogisticRegression(max_iter=500)
        logreg.fit(source_emb_arr[source_taxi_dropoff_mask], source_taxi_dropoff_label)
        target_taxi_dropoff_pred = logreg.predict(target_emb_arr[target_taxi_dropoff_mask])
        target_taxi_dropoff_acc = (target_taxi_dropoff_pred == target_taxi_dropoff_label).mean()
        print("taxi pickup acc %.4f, taxi dropoff acc %.4f" % (target_taxi_pickup_acc, target_taxi_dropoff_acc))
        da_result += target_taxi_pickup_acc
        da_result += target_taxi_dropoff_acc
    
    if tcity != 'Porto':
        source_bike_pickup_mask = source_transport['bike_pickup'][0]
        source_bike_pickup_label = source_transport['bike_pickup'][1]
        source_bike_dropoff_mask = source_transport['bike_dropoff'][0]
        source_bike_dropoff_label = source_transport['bike_dropoff'][1]
        target_bike_pickup_mask = target_transport['bike_pickup'][0]
        target_bike_pickup_label = target_transport['bike_pickup'][1]
        target_bike_dropoff_mask = target_transport['bike_dropoff'][0]
        target_bike_dropoff_label = target_transport['bike_dropoff'][1]

        logreg = LogisticRegression(max_iter=500)
        logreg.fit(source_emb_arr[source_bike_pickup_mask], source_bike_pickup_label)
        target_bike_pickup_pred = logreg.predict(target_emb_arr[target_bike_pickup_mask])
        target_bike_pickup_acc = (target_bike_pickup_pred == target_bike_pickup_label).mean()
        
        logreg = LogisticRegression(max_iter=500)
        logreg.fit(source_emb_arr[source_bike_dropoff_mask], source_bike_dropoff_label)
        target_bike_dropoff_pred = logreg.predict(target_emb_arr[target_bike_dropoff_mask])
        target_bike_dropoff_acc = (target_bike_dropoff_pred == target_bike_dropoff_label).mean()
        print("bike pickup acc %.4f, bike dropoff acc %.4f" % (target_bike_pickup_acc, target_bike_dropoff_acc))
        da_result += target_bike_pickup_acc
        da_result += target_bike_dropoff_acc
    # save embeddings
    if da_result > best_result:
        best_result = da_result
        np.save(save_path + "%s.npy" % (scity), arr = source_emb_arr)
        np.save(save_path + "%s.npy" % (tcity), arr = target_emb_arr)
        print("Saving embeddings ......")
        
    print()

with open(save_path + 'best_result', 'w') as outfile:
    outfile.write(str(best_result) + '\n')
    print("Current best result %.4f" % best_result)
    print("Using args %s" % str(args))
    if best_result > previous_best:
        outfile.write("Using args %s" % str(args))
    else:
        outfile.write(lines[1])
