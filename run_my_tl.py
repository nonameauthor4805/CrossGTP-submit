import numpy as np
import argparse
from model import *
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import os
import time
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--scity', type=str, default='NY', help='Source city')
parser.add_argument('--tcity', type=str, default='DC', help='Target city')
parser.add_argument('--dataname', type=str, default='Taxi', help = 'Within bike or taxi')
parser.add_argument('--datatype', type=str, default='pickup', help = 'Within pickup or dropoff')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--model', type=str, default='STNet')
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--num_spatial", type=int, default=3, help='number of spatial layers in the model')
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=-1, help='Random seed. -1 means do not manually set. ')
parser.add_argument("--data_amount", type=int, default=0, help='0: full data (8months), 30: month data, 7: week data, 3: 3 day data, 1: 1day data')
parser.add_argument("--source_model", type=str, default =None, help = 'file name for the pretrained model')
parser.add_argument('--embedding_name', type=str, default=None, help='file name for region embeddings')
parser.add_argument('--topk', type=int, default=10, help='topk for constructing the cross city graph')
parser.add_argument('--mmd_weight', type=float, default=0.1, help='mmd for source/target output features')
args = parser.parse_args()

if args.seed != -1:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
gpu_available = torch.cuda.is_available()
if gpu_available:
    device = torch.device("cuda:0")
else:
    device = torch.device('cpu')

dataname = args.dataname
scity = args.scity 
tcity = args.tcity 
datatype = args.datatype
batch_size = args.batch_size
learning_rate = args.learning_rate
weight_decay = args.weight_decay
num_conv = args.num_spatial
num_epochs = args.num_epochs
mmd_weight = args.mmd_weight

if args.source_model is not None:
    source_path = '../saved_models/%s/%s/%s/%s_%s.pt' % \
    (scity, dataname, datatype, args.model, args.source_model)
    print("Load model from ", source_path)

start_time = time.time()
print("Running CrossGTP, from %s to %s, %s %s experiments, with %d data, %s model" % (scity, tcity, dataname, datatype, args.data_amount, args.model))

target_data = np.load("../data/%s/%s%s_%s.npy" % (tcity, dataname, tcity, datatype))
lng_target = target_data.shape[1]
lat_target = target_data.shape[2]
spatial_mask_target = target_data.sum(0) > 0
spatial_mask_target = spatial_mask_target.reshape(1, lng_target, lat_target)
th_spatial_mask_target = torch.Tensor(spatial_mask_target).to(device)
print("Valid regions target: %d" % np.sum(spatial_mask_target))

source_data = np.load("../data/%s/%s%s_%s.npy" % (scity, dataname, scity, datatype))
spatial_mask_source = source_data.sum(0) > 0
lng_source = source_data.shape[1]
lat_source = source_data.shape[2]
spatial_mask_source = spatial_mask_source.reshape(1, lng_source, lat_source)
th_spatial_mask_source = torch.Tensor(spatial_mask_source).to(device)
print("Valid regions source: %d" % np.sum(spatial_mask_source))

# compute bias for date 
"""
bias = 0
if scity == 'CHI':
    if tcity == 'DC':
        bias = -7 * 24
    elif tcity == 'BOS':
        bias = -6 * 24
    elif tcity == 'Porto':
        bias = -5 * 24
elif scity == 'NY':
    if tcity == 'DC':
        if dataname == 'Bike':
            bias = -7 * 24
        elif dataname == 'Taxi':
            bias = -5 * 24
    elif tcity == 'Porto':
        bias = -3 * 24
    elif tcity == 'BOS':
        bias = -6 * 24
"""
# measure the weekday of starting
# 0: sunday
# 1: monday
# etc
start_days = {
    ("NY", "Taxi"): 3, 
    ("NY", "Bike"): 5, 
    ("CHI", "Taxi"): 5, 
    ("CHI", "Bike"): 5, 
    ("DC", "Bike"): 5, 
    ("DC", "Taxi"): 5, 
    ("BOS", "Bike"): 0, 
}
source_startday = start_days[(scity, dataname)]
target_startday = start_days[(tcity, dataname)]

lag = [-6, -5, -4, -3, -2, -1]
source_data, _, _ = min_max_normalize(source_data)
# let source data to start from a sunday
source_data = source_data[(7-source_startday) * 24:, :, :]
target_data, max_val, min_val = min_max_normalize(target_data)
# transform to [-1, 1] normalize
# source_data = (source_data - 0.5) * 2
# target_data = (target_data - 0.5) * 2

# transform to 
target_train_x, target_train_y, target_train_hours, target_train_weekdays, \
    val_x, val_y, val_hours, val_weekdays, test_x, test_y, test_hours, test_weekdays = split_x_y_wdayhour(target_data, lag, target_startday)
# target_train_x, target_train_y, target_train_hours,\
#     val_x, val_y, val_hours, test_x, test_y, test_hours = split_x_y_whour(target_data, lag)
if args.data_amount != 0:
    target_train_x = target_train_x[-args.data_amount * 24:, :, :, :]
    target_train_y = target_train_y[-args.data_amount * 24:, :, :, :]
    target_train_hours = target_train_hours[-args.data_amount * 24:]
    target_train_weekdays = target_train_weekdays[-args.data_amount * 24:]
print("Split to: ")
print("target train_x: %s, target train_y: %s" % (str(target_train_x.shape), str(target_train_y.shape)))
print("val_x %s, val_y %s" % (str(val_x.shape), str(val_y.shape)))
print("test_x %s, test_y %s" % (str(test_x.shape), str(test_y.shape)))

source_hour_lag = 1
source_sample_days = 1
num_source_days = source_data.shape[0] // 24
num_source_weeks = num_source_days // 7

def sample_source_data(hours, days):
    sampled_sources = [[] for _ in range(2 * source_sample_days + 1)]
    # print(hours)
    for (h, d) in zip(hours, days):
        week = random.randint(1, num_source_weeks - 2)
        for i in range(2 * source_sample_days + 1):
            idx = (week * 7 + d - 1 + i) * 24 + h - source_hour_lag
            
            sampled_sources[i].append(source_data[idx:idx + 2 * source_hour_lag + 1])
        # print()
    return [np.stack(sampled_sources[i], axis = 0) for i in range(2 * source_sample_days + 1)]


source_embs = np.load("../embeddings/%s_%s/%s.npy" % (scity, tcity, scity))
target_embs = np.load("../embeddings/%s_%s/%s.npy" % (scity, tcity, tcity))
print("Source embs:", source_embs.shape)
print("Target embs:", target_embs.shape)

source_net = STNet(1, num_conv, th_spatial_mask_source).to(device)
target_net = CrossGTPNet(1, num_conv, source_embs, target_embs, th_spatial_mask_target, topk = args.topk).to(device)
target_net.crosscity_graph = target_net.crosscity_graph.to(device)
if args.source_model is not None:
    print("Load model from %s." % source_path)
    source_dict = torch.load(source_path)
    source_net.load_state_dict(source_dict)
    loaded_params = 0
    for k in source_dict:
        if k in target_net.state_dict():
            target_net.state_dict()[k].copy_(source_dict[k].data)
            loaded_params += 1
    print("Loaded %d parameters out of %d from %s" % (loaded_params, len(target_net.state_dict()), source_path))
best_val_rmse = 999 
best_test_rmse = 999
best_test_mae = 999
param_list = list(source_net.parameters()) + list(target_net.parameters())
optimizer = optim.Adam(param_list, lr = learning_rate, weight_decay = args.weight_decay)

def evaluate(source_net_, target_net_, Xs, Ys, hours, days):
    with torch.no_grad():
        se = 0
        ae = 0
        valid_points = 0
        num_samples = Xs.shape[0]
        idx = 0
        while 1:
            if idx + batch_size < num_samples:
                batch_X = Xs[idx:idx + batch_size, :, :, :]
                batch_y = Ys[idx:idx + batch_size, :, :, :]
                batch_hour = hours[idx:idx + batch_size]
                batch_day = days[idx:idx + batch_size]
            else:
                batch_X = Xs[idx:, :, :, :]
                batch_y = Ys[idx:, :, :, :]
                batch_hour = hours[idx:]
                batch_day = days[idx:idx + batch_size]
            idx += batch_size
            batch_source_x = sample_source_data(batch_hour, batch_day)
            batch_X = torch.Tensor(batch_X).to(device)
            batch_y = torch.Tensor(batch_y).to(device)
            for i in range(len(batch_source_x)):
                batch_source_x[i] = torch.Tensor(batch_source_x[i]).to(device)
            # implement forward of crossgtp model
            source_feats = []
            for i in range(len(batch_source_x)):
                source_feat, _ = source_net_(batch_source_x[i], return_feat = True)
                source_feats.append(source_feat)
            # out_, _ = target_net_(batch_X, source_feat)
            out_ = target_net_(batch_X, source_feats)
            # print("out_", out_.shape)
            cur_batch_size = batch_X.shape[0]
            lag = batch_y.shape[1]
            batch_y = batch_y.view(cur_batch_size, lag, -1)[:, :, th_spatial_mask_target.view(-1).bool()]
            # print("batch_y", batch_y.shape)
            valid_points += cur_batch_size * spatial_mask_target.sum()
            se += ((out_ - batch_y) ** 2).sum().item()
            ae += (out_ - batch_y).abs().sum().item()
            if idx >= num_samples:
                break
    return np.sqrt(se / valid_points), ae / valid_points


def train_epoch(source_net_, target_net_, Xs, Ys, hours, days, optimizer):
    epoch_loss = []
    epoch_mmd_loss = []
    num_samples = Xs.shape[0]
    permute_idx = np.random.permutation(num_samples)
    Xs = Xs[permute_idx, :, :, :]
    Ys = Ys[permute_idx, :, :, :]
    hours = hours[permute_idx]
    idx = 0
    while 1:
        if idx + batch_size < num_samples:
            batch_X = Xs[idx:idx + batch_size, :, :, :]
            batch_y = Ys[idx:idx + batch_size, :, :, :]
            batch_hour = hours[idx:idx + batch_size]
            batch_day = days[idx:idx+batch_size]
        else:
            batch_X = Xs[idx:, :, :, :]
            batch_y = Ys[idx:, :, :, :]
            batch_hour = hours[idx:]
            batch_day = days[idx:idx+batch_size]
        idx += batch_size
        batch_source_x = sample_source_data(batch_hour, batch_day)
        batch_X = torch.Tensor(batch_X).to(device)
        batch_y = torch.Tensor(batch_y).to(device)
        for i in range(len(batch_source_x)):
            batch_source_x[i] = torch.Tensor(batch_source_x[i]).to(device)
        # print('batch_X', batch_X.shape)
        # print("batch_y", batch_y.shape)
        # print('batch_source_x', batch_source_x.shape)
        source_feats = []
        for i in range(len(batch_source_x)):
            source_feat, _ = source_net_(batch_source_x[i], return_feat = True)
            source_feats.append(source_feat)
        # print('source_feat', source_feat.shape)
        # implement forward of crossgtp model 
        # out_, mmd = target_net_(batch_X, source_feat)
        if args.mmd_weight != 0:
            out_, mmd = target_net_(batch_X, source_feats)
        else:
            out_ = target_net_(batch_X, source_feats)
        # print("out_", out_.shape)
        lag = batch_y.shape[1]
        cur_batch_size = batch_y.shape[0]
        target_y = batch_y.view(cur_batch_size, lag, -1)[:, :, th_spatial_mask_target.view(-1).bool()]
        loss_pred = ((out_ - target_y) ** 2).sum()
        if args.mmd_weight != 0:
            loss = loss_pred + mmd * mmd_weight
        else:
            loss = loss_pred
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_list, 5)
        optimizer.step()
        epoch_loss.append(loss_pred.item())
        if args.mmd_weight != 0:
            epoch_mmd_loss.append(mmd.item())
        if idx + batch_size >= num_samples:
            break
    if args.mmd_weight != 0:
        return np.mean(epoch_loss), np.mean(epoch_mmd_loss)
    else:
        return np.mean(epoch_loss)

for ep in range(num_epochs):
    source_net.train()
    target_net.train()
    if args.mmd_weight != 0:
        avg_loss, avg_mmd_loss = train_epoch(source_net, target_net, target_train_x, target_train_y, target_train_hours, target_train_weekdays, optimizer)
        print("[%.2fs]Epoch %d, average pred loss %.4f, mmd loss %.4f" % (time.time() - start_time, ep, avg_loss, avg_mmd_loss))
    else:
        avg_loss = train_epoch(source_net, target_net, target_train_x, target_train_y, target_train_hours, target_train_weekdays, optimizer)
        print("[%.2fs]Epoch %d, average pred loss %.4f" % (time.time() - start_time, ep, avg_loss))
    source_net.eval()
    target_net.eval()
    rmse_vals = []
    mae_vals = []
    rmse_tests = []
    mae_tests = []
    # for i in range(3):
    rmse_val, mae_val = evaluate(source_net, target_net, val_x, val_y, val_hours, val_weekdays)
    rmse_test, mae_test = evaluate(source_net, target_net, test_x, test_y, test_hours, test_weekdays)
    # rmse_vals.append(rmse_val)
    # mae_vals.append(mae_val)
    # rmse_tests.append(rmse_test)
    # mae_tests.append(mae_test)
    
    if rmse_val < best_val_rmse:
        best_val_rmse =  rmse_val
        best_test_rmse = rmse_test
        best_test_mae = mae_test
        print("Update best test...")
    """
    if np.mean(rmse_vals) < best_val_rmse:
        best_val_rmse = np.mean(rmse_vals)
        best_test_rmse = np.mean(rmse_tests)
        best_test_mae = np.mean(mae_tests)
        print("Update best test ...")
        print("RMSE vals: %s" % str(rmse_vals))
        print("RMSE tests: %s" % str(rmse_tests))
    """
    print("validation rmse %.4f, mae %.4f" % (rmse_val * (max_val - min_val), mae_val * (max_val - min_val)))
    print("test rmse %.4f, mae %.4f" % (rmse_test * (max_val - min_val), mae_test * (max_val - min_val)))
    print()
print("Best test rmse %.4f, mae %.4f" % (best_test_rmse * (max_val - min_val), best_test_mae * (max_val - min_val)))
