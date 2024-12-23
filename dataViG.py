import os
import time
import glob
import copy
import random
import pickle
import argparse
import itertools
import gzip
import heapq

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from einops.layers.torch import Rearrange

import dgl

import multiprocessing as mp

REALTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class imageSet(torch.utils.data.Dataset):
    def __init__(self, filenames, crop=None, size=(256, 256), stride=256):
        super().__init__()

        self._filenames = filenames
        self._crop = crop
        self._size = size
    
        self._graphs = []
        self._images = []
        self._labels = []
        assert self._filenames[-4:] == ".pkl"
        print("[ImageSet]: Pre-loading data......")
        try:
            with open(self._filenames, "rb") as fin: 
                self._images, self._labels, *_ = pickle.load(fin)
        except:
            fin = gzip.GzipFile(self._filenames, "rb")
            self._images, self._labels, *_ = pickle.load(fin)
            
        print("[ImageSet]: Interpolating features......")
        pop_idx = []
        for idx in tqdm(range(len(self._images))): 
            image = self._images[idx]
            label = self._labels[idx]
            if image is None:
                pop_idx.append(idx)
                continue
            self._images[idx] = F.interpolate(image[None, :3, :, :], size=self._size, 
                                              mode='bilinear', align_corners=True)[0]
            self._labels[idx] = F.interpolate(label[None, :, :, :], size=self._size, 
                                              mode='bilinear', align_corners=True)[0]
        print(pop_idx)
        for idx in sorted(pop_idx, reverse=True):
            del self._images[idx]
            del self._labels[idx]
        assert len(self._images) == len(self._labels)
        print("[ImageSet]: Cropping shapes......")
        indices = []
        ranges = []
        for index, image in enumerate(tqdm(self._images)): 
            if image.shape[1] <= size[0] and image.shape[2] <= size[1]: 
                indices.append(index)
                ranges.append([0, 0, image.shape[1], image.shape[2]])
            else: 
                for beginX in range(0, image.shape[1], stride): 
                    for beginY in range(0, image.shape[2], stride): 
                        stepX = min(size[0], image.shape[1] - beginX)
                        stepY = min(size[1], image.shape[2] - beginY)
                        indices.append(index)
                        ranges.append([beginX, beginY, beginX + stepX, beginY + stepY])
        self._indices = indices
        self._ranges = ranges
        print(f"[ImageSet]: Dataset analyzed, totally {len(indices)} images")
        

    def __getitem__(self, index): 
        image, label = self._load(self._indices[index], self._ranges[index])
        return image, label

    def __len__(self): 
        return len(self._indices)
        
    def _load(self, index, crop, dtype=REALTYPE, device=DEVICE): 
        image = self._images[index] 
        label = self._labels[index]
        bins = [label.shape[-2], label.shape[-1]]
        image = image[:, crop[0]:crop[2], crop[1]:crop[3]]
        label = label[:, crop[0]:crop[2], crop[1]:crop[3]]
        beginX = 0
        beginY = 0
        endX = self._size[0]
        endY = self._size[1]
        diffX = crop[2] - crop[0]
        diffY = crop[3] - crop[1]
        if diffX < self._size[0]:
            endX = beginX + diffX
        if diffY < self._size[1]: 
            endY = beginY + diffY
        x = torch.zeros(image.shape[:1] + self._size, dtype=dtype, device=device)
        y = torch.zeros(label.shape[:1] + self._size, dtype=dtype, device=device)
        x[:, beginX:endX, beginY:endY] = image
        y[:, beginX:endX, beginY:endY] = label
        
        
        return x, y 


class ViGSet(torch.utils.data.Dataset): 
    def __init__(self, filenames, crop=None, size=(256, 256), stride=256):
        super().__init__()

        self._filenames = filenames
        self._crop = crop
        self._size = size
    
        self._graphs = []
        self._images = []
        self._labels = []
        self._e_ccs = []
        self._e_nns = []
        self._w_nns = []
        self._v_ns = []
        self._e_cns = []
        assert self._filenames[-4:] == ".pkl"
        print("[ImageSet]: Pre-loading data......")
        try:
            with open(self._filenames, "rb") as fin: 
                self._images, self._labels, self._e_ccs, self._e_nns, self._w_nns, self._v_ns, self._e_cns = pickle.load(fin)
        except:
            fin = gzip.GzipFile(self._filenames, "rb")
            self._images, self._labels, self._e_ccs, self._e_nns, self._w_nns, self._v_ns, self._e_cns = pickle.load(fin)
            
        print("[ImageSet]: Interpolating features......")
        pop_idx = []
        for idx in tqdm(range(len(self._images))): 
            image = self._images[idx]
            label = self._labels[idx]
            if image is None:
                pop_idx.append(idx)
                continue
            self._images[idx] = F.interpolate(image[None, :, :, :], size=self._size, 
                                              mode='bilinear', align_corners=True)[0]
            self._labels[idx] = F.interpolate(label[None, :2, :, :], size=self._size, 
                                              mode='bilinear', align_corners=True)[0]

            cl_label = torch.round(self._labels[idx]*5)
            self._labels[idx] = torch.cat((self._labels[idx], cl_label), dim=0)
            gnet_mask = (self._e_nns[idx][0] < self._size[0]*3//8) & (self._e_nns[idx][1] < self._size[1]*3//8)
            self._e_nns[idx] = self._e_nns[idx][:,gnet_mask]
            self._w_nns[idx] = self._w_nns[idx][gnet_mask]

            e_cn_mask_0 = self._e_cns[idx][0][1] < self._size[1]*3//8      
            self._e_cns[idx][0] = self._e_cns[idx][0][:,e_cn_mask_0]

            e_cn_mask_1 = self._e_cns[idx][1][1] < self._size[1]*3//8       
            self._e_cns[idx][1] = self._e_cns[idx][1][:,e_cn_mask_1]

            e_cn_mask_2 = self._e_cns[idx][2][1] < self._size[1]*3//8      
            self._e_cns[idx][2] = self._e_cns[idx][2][:,e_cn_mask_2]

            e_cn_mask_3 = self._e_cns[idx][3][1] < self._size[1]*3//8      
            self._e_cns[idx][3] = self._e_cns[idx][3][:,e_cn_mask_3]

            self._v_ns[idx] = self._v_ns[idx][:self._size[1]*3//8]

            
        for idx in sorted(pop_idx, reverse=True):
            del self._images[idx]
            del self._labels[idx]
            del self._e_ccs[idx]
            del self._e_nns[idx]
            del self._w_nns[idx]
            del self._v_ns[idx]
            del self._e_cns[idx]
        assert len(self._images) == len(self._labels)
        print("[ImageSet]: Cropping shapes......")
        indices = []
        ranges = []
        for index, image in enumerate(tqdm(self._images)): 
            if image.shape[1] <= size[0] and image.shape[2] <= size[1]: 
                indices.append(index)
                ranges.append([0, 0, image.shape[1], image.shape[2]])
            else: 
                for beginX in range(0, image.shape[1], stride): 
                    for beginY in range(0, image.shape[2], stride): 
                        stepX = min(size[0], image.shape[1] - beginX)
                        stepY = min(size[1], image.shape[2] - beginY)
                        indices.append(index)
                        ranges.append([beginX, beginY, beginX + stepX, beginY + stepY])
        self._indices = indices
        self._ranges = ranges
        print(f"[ImageSet]: Dataset analyzed, totally {len(indices)} images")

    def __getitem__(self, index): 
        hg, label = self._load(self._indices[index], self._ranges[index])
        return hg, label

    def __len__(self): 
        return len(self._indices)
        
    def _load(self, index, crop, dtype=REALTYPE, device=DEVICE): 
        image = self._images[index] 
        label = self._labels[index]
        bins = [label.shape[-2], label.shape[-1]]
        image = image[:, crop[0]:crop[2], crop[1]:crop[3]]
        label = label[:, crop[0]:crop[2], crop[1]:crop[3]]
        beginX = 0
        beginY = 0
        endX = self._size[0]
        endY = self._size[1]
        diffX = crop[2] - crop[0]
        diffY = crop[3] - crop[1]
        if diffX < self._size[0]: 
            endX = beginX + diffX
        if diffY < self._size[1]: 
            endY = beginY + diffY
        x = torch.zeros(image.shape[:1] + self._size, dtype=dtype, device='cpu')
        y = torch.zeros(label.shape[:1] + self._size, dtype=dtype, device='cpu')
        x[:, beginX:endX, beginY:endY] = image
        y[:, beginX:endX, beginY:endY] = label
    
        e_ccs = copy.deepcopy(self._e_ccs[index])
        e_nn = copy.deepcopy(self._e_nns[index])
        w_nn = copy.deepcopy(self._w_nns[index])
        v_n = copy.deepcopy(self._v_ns[index])
        e_cns = copy.deepcopy(self._e_cns[index]) 
        
        scale = 4
        meshX, meshY = self._size
        bins[0] = bins[0] // scale
        bins[1] = bins[1] // scale
        for idx in range(len(e_ccs)):
            if idx == 0:
                e_ccs[idx], e_cns[idx], drop_net_list = graph_crop(e_ccs[idx], e_cns[idx], bins, meshX//scale, meshY//scale, list(map(lambda x: x//scale, crop)))
            else:
                e_ccs[idx], e_cns[idx], _ = graph_crop(e_ccs[idx], e_cns[idx], bins, meshX//scale, meshY//scale, list(map(lambda x: x//scale, crop)))
            bins[0] = bins[0] // 2
            bins[1] = bins[1] // 2
            scale *= 2
        #e_nn = nn_crop(e_nn, e_cns[0])
        ##### TO generate heteo-graph
        scale = 4
        bins = [x.shape[-2] // scale, x.shape[-1] // scale]
        scale *= 2
        bins_x2 = [x.shape[-2] // scale, x.shape[-1] // scale]
        scale *= 2
        bins_x4 = [x.shape[-2] // scale, x.shape[-1] // scale] 
        scale *= 2
        bins_x8 = [x.shape[-2] // scale, x.shape[-1] // scale]     

        if e_nn.shape[0] == 0:
            num_gnet = 0
        else:
            num_gnet = e_nn.max() + 1
        for i in range(4):
            e_ccs[i] = e_ccs[i].tolist()
            e_cns[i] = e_cns[i].tolist()
        hg = dgl.heterograph({
                ('Gcell', 'cc', 'Gcell'): (e_ccs[0][0], e_ccs[0][1]) if len(e_ccs[0]) != 0 else ([], []),
                ('Gcell', 'cn', 'Gnet'): (e_cns[0][0], e_cns[0][1]) if len(e_cns[0]) != 0 else ([], []),
                ('Gnet', 'nc', 'Gcell'): (e_cns[0][1], e_cns[0][0]) if len(e_cns[0]) != 0 else ([], []),
                ('Gnet', 'nn', 'Gnet'): (e_nn[0], e_nn[1])  if e_nn.shape[0] != 0 else ([], []),
                
                ('Gcellx2', 'ccx2', 'Gcellx2'): (e_ccs[1][0], e_ccs[1][1]) if len(e_ccs[1]) != 0 else ([], []),
                ('Gcellx2', 'cnx2', 'Gnet'): (e_cns[1][0], e_cns[1][1]) if len(e_cns[1]) != 0 else ([], []),
                ('Gnet', 'ncx2', 'Gcellx2'): (e_cns[1][1], e_cns[1][0]) if len(e_cns[1]) != 0 else ([], []),
                
                ('Gcellx4', 'ccx4', 'Gcellx4'): (e_ccs[2][0], e_ccs[2][1]) if len(e_ccs[2]) != 0 else ([], []),
                ('Gcellx4', 'cnx4', 'Gnet'): (e_cns[2][0], e_cns[2][1]) if len(e_cns[2]) != 0 else ([], []),
                ('Gnet', 'ncx4', 'Gcellx4'): (e_cns[2][1], e_cns[2][0]) if len(e_cns[2]) != 0 else ([], []),
                
                ('Gcellx8', 'ccx8', 'Gcellx8'): (e_ccs[3][0], e_ccs[3][1]) if len(e_ccs[3]) != 0 else ([], []),
                ('Gcellx8', 'cnx8', 'Gnet'): (e_cns[3][0], e_cns[3][1]) if len(e_cns[3]) != 0 else ([], []),
                ('Gnet', 'ncx8', 'Gcellx8'): (e_cns[3][1], e_cns[3][0]) if len(e_cns[3]) != 0 else ([], []),
                },
                {'Gcell': bins[0]*bins[1], 'Gnet': num_gnet, 'Gcellx2': bins_x2[0]*bins_x2[1], 'Gcellx4': bins_x4[0]*bins_x4[1], 'Gcellx8': bins_x8[0]*bins_x8[1]}
            )
        hg = dgl.add_self_loop(hg, etype='cc')
        hg = dgl.add_self_loop(hg, etype='ccx2')
        hg = dgl.add_self_loop(hg, etype='ccx4')
        hg = dgl.add_self_loop(hg, etype='ccx8')
        rearrange_x = Rearrange('c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 4, p2 = 4)
        x = rearrange_x(x)
        hg.ndata['h'] = {'Gcell':x, 'Gnet':torch.tensor(v_n, dtype=dtype)}
        hg.edata['h'] = {'nn':torch.tensor(w_nn, dtype=dtype)}
        hg = dgl.remove_nodes(hg, torch.tensor(drop_net_list, dtype=torch.int64), ntype='Gnet')
        if hg.num_nodes(ntype='Gnet') > 64*64:
            hg = dgl.remove_nodes(hg, torch.randint(0, hg.num_nodes(ntype='Gnet'), (1, hg.num_nodes(ntype='Gnet')-64*64))[0], ntype='Gnet')

        
        return hg, y




def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def get_top_n_bbox(bbox_list, n=10000):
    indexed_areas = [(calculate_area(bbox), i) for i, bbox in enumerate(bbox_list)]
    indexed_areas.sort(key=lambda x: x[1], reverse=True)
    top_n_indices = [i for i, area in indexed_areas[:n]]
    top_n_bboxes = [bbox_list[i] for i in top_n_indices]
    return top_n_bboxes


def cal_macro_margin(macro, graph, layout, dtype=REALTYPE, device=DEVICE):
    nodes = graph['nodes']
    macro_nodes = list(filter(lambda x: x['macro'] == True, nodes))
    macro_h = np.zeros(macro.shape)
    macro_v = np.zeros(macro.shape)
    for x in range(macro.shape[0]):
        for y in range(macro.shape[1]):
            if macro[x, y] == 1:
                macro_h[x][y] = 0
                macro_v[x][y] = 0
                continue
            left_index = np.where(macro[x,:y] == 1)
            if left_index[0].shape[0] == 0:
                left = 0
            else:
                left = left_index[0][-1]
            right_index = np.where(macro[x,y+1:] == 1)
            if right_index[0].shape[0] == 0:
                right = macro.shape[1]
            else:
                right = right_index[0][0] + y + 1
            down_index = np.where(macro[:x,y] == 1)
            if down_index[0].shape[0] == 0:
                down = 0
            else:
                down = down_index[0][-1]
            top_index = np.where(macro[x+1:,y] == 1)
            if top_index[0].shape[0] == 0:
                top = macro.shape[0]
            else:
                top = top_index[0][0] + x + 1
            macro_h[x][y] = 1 - (right - left)/macro.shape[1] if right != macro.shape[1] or left != 0 else 0
            macro_v[x][y] = 1 - (top - down)/macro.shape[0] if top != macro.shape[0] or down != 0 else 0
    for node in macro_nodes:
        x1 = int((node['posX'] - layout[0])/(layout[2] - layout[0])*macro.shape[0])
        x2 = int((node['posX'] + node['sizeX'] - layout[0])/(layout[2] - layout[0])*macro.shape[0]) + 1
        y1 = int((node['posY'] - layout[1])/(layout[3] - layout[1])*macro.shape[1])
        y2 = int((node['posY'] + node['sizeY'] - layout[1])/(layout[3] - layout[1])*macro.shape[1]) + 1
        x1 = macro.shape[0] - 1 if x1 > macro.shape[0] - 1 else x1
        x2 = macro.shape[0] - 1 if x2 > macro.shape[0] - 1 else x2
        y1 = macro.shape[1] - 1 if y1 > macro.shape[1] - 1 else y1
        y2 = macro.shape[1] - 1 if y2 > macro.shape[1] - 1 else y2
        for i in range(x1, x2 + 1):
            for j in range(0, 3):
                macro_h[i][y1+j] = 1
                macro_v[i][y1+j] = 1
                macro_h[i][y2-j] = 1
                macro_v[i][y2-j] = 1                         
        for j in range(y1, y2 + 1):
            for i in range(0, 3):
                macro_h[x1+i][j] = 1
                macro_v[x1+i][j] = 1
                macro_h[x2-i][j] = 1
                macro_v[x2-i][j] = 1
    return macro_h, macro_v
   
def cal_overlap(bbox1, bbox2, v_n=None, mode='inter', num_former=0):
    '''
    mode = 'inter' / 'intra1' / 'intra2'
    '''
    A = bbox1.shape[0]
    B = bbox2.shape[0]
    xy_max = np.minimum(bbox1[:, np.newaxis, 2:].repeat(B, axis=1),
                        np.broadcast_to(bbox2[:,2:], (A, B, 2)))
    xy_min = np.maximum(bbox1[:, np.newaxis, :2].repeat(B, axis=1),
                        np.broadcast_to(bbox2[:,:2], (A, B, 2)))
    inter = np.clip(xy_max - xy_min + 1, a_min=0, a_max=np.inf)
    area = inter[:, :, 0] * inter[:, :, 1]
    idx = area.nonzero()
    area = area[idx]
    idx = np.array(idx)
    if mode == 'inter':
        idx[1] = idx[1] + A
    elif mode == 'intra2':
        idx = idx + num_former
    idx_reverse = idx[[1,0],:]
    idx = np.concatenate((idx, idx_reverse), axis=1)
    area = np.concatenate((area, area))
    return idx, area
    
def gen_vig_data(data, scale=4):
    graph = data['graph']
    bins = list(data['congH'].shape)
    bins[0] = bins[0] // scale
    bins[1] = bins[1] // scale
    canvas = data['range']
    x0, y0, x1, y1 = canvas
    binSizeX = (x1 - x0)/bins[0]
    binSizeY = (y1 - y0)/bins[1]
    nodes = graph["nodes"]
    nets = sorted(graph["nets"],key=len)
    bbox_list = []
    # edge_pair_sorted = []
    n_nets = len(nets)
    for idx, net in enumerate(nets):
        if len(net) > 16:
            break #sorted
        min_x_list = []
        max_x_list = []
        min_y_list = []
        max_y_list = []
        bin_index = []
        for jdx in range(len(net)):
            posx0 = nodes[net[jdx]]['posX']
            posy0 = nodes[net[jdx]]['posY']
            sizeX = nodes[net[jdx]]['sizeX']
            sizeY = nodes[net[jdx]]['sizeY']
            posx1 = posx0 + sizeX
            posy1 = posy0 + sizeY
            x_index0 = min(int(posx0/binSizeX),bins[0]-1)
            x_index1 = min(int(posx1/binSizeX),bins[0]-1)
            y_index0 = min(int(posy0/binSizeY),bins[1]-1)
            y_index1 = min(int(posy1/binSizeY),bins[1]-1)
            bin_index_t = (x_index0+x_index1)*bins[1]//2 + (y_index0+y_index1)//2
            bin_index.append(bin_index_t)
            min_x_list.append(x_index0)
            max_x_list.append(x_index1)
            min_y_list.append(y_index0)
            max_y_list.append(y_index1)
        min_x = min(min_x_list)
        max_x = max(max_x_list)
        min_y = min(min_y_list)
        max_y = max(max_y_list)
        bin_index = list(set(bin_index))
        if bins[0]*bins[1] >= 32*32 and ((max_x - min_x + 1)*(max_y - min_y + 1) > bins[0]*bins[1]*0.25/100 or len(bin_index)/((max_x - min_x + 1)*(max_y - min_y + 1)) < 0.25 or len(bin_index) < 2):
            continue
        elif bins[0]*bins[1] < 32*32 and ((max_x - min_x + 1)*(max_y - min_y + 1) > bins[0]*bins[1]*0.5/100 or len(bin_index)/((max_x - min_x + 1)*(max_y - min_y + 1)) < 0.5 or len(bin_index) < 2):
            continue
        else:
            bbox_list.append((min_x, min_y, max_x, max_y))

    bbox_list = list(set(bbox_list))
    bbox_list = get_top_n_bbox(bbox_list, int(n_nets*0.01))
    num_hyperedge = len(bbox_list)

    v_n = np.zeros((num_hyperedge, 3))
    source_index = [] # edge_index (GCell --> GNet)
    target_index = []
    edge_cn_index = [] 
    for i in range(num_hyperedge):
        min_x, min_y, max_x, max_y = bbox_list[i]
        span_h = max_x - min_x + 1
        span_v = max_y - min_y + 1
        area = span_h * span_v
        v_n[i][0] = span_h
        v_n[i][1] = span_v
        v_n[i][2] = area
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                source_index.append(x*bins[1]+y)
                target_index.append(i)
    edge_cn_index.append(source_index)
    edge_cn_index.append(target_index)
    edge_cn_index = np.array(edge_cn_index)
    bbox_list = np.array(bbox_list)
    n = bbox_list.shape[0]
    bbox1 = bbox_list[:n//2]
    v_n1 = v_n[:n//2,2]
    bbox2 = bbox_list[n//2:]
    v_n2 = v_n[n//2:,2] 
    idx1, area1 = cal_overlap(bbox1, bbox2, mode='inter')
    idx2, area2 = cal_overlap(bbox1, bbox1, v_n1, mode='intra1')
    idx3, area3 = cal_overlap(bbox2, bbox2, v_n2, mode='intra2', num_former=bbox1.shape[0])
    overlapping_pairs = np.concatenate((idx1, idx2, idx3), axis=-1)
    overlapping_area = np.concatenate((area1, area2, area3))
    source_node = []
    target_node = []
    edge_index = []
    for i in range(0, bins[0]):
        for j in range(0, bins[1]):
            if(j < bins[1] - 1):
                source_node.append(i*bins[1]+j)
                target_node.append(i*bins[1]+j+1)
            if(i < bins[0] - 1):
                source_node.append(i*bins[1]+j)
                target_node.append((i+1)*bins[1]+j)
    edge_index.append(source_node)
    edge_index.append(target_node)
    edge_index = np.array(edge_index)
    edge_index_rev = edge_index[[1,0],:]
    edge_index = np.concatenate((edge_index, edge_index_rev),axis=1)

    return edge_index, overlapping_pairs, overlapping_area, v_n, edge_cn_index, bins   

def downsample(e_cc, e_cn, bins, scale=2):
    ds_bins = bins
    ds_bins[0] = bins[0] // scale
    ds_bins[1] = bins[1] // scale
    e_cc = (e_cc / bins[1]) // scale * ds_bins[1] + e_cc % bins[1] // scale
    e_cn[0] = (e_cn[0] / bins[1]) // scale * ds_bins[1] + e_cn[0] % bins[1] // scale
    e_cc = e_cc.T
    e_cc = np.array(list(set([tuple(t) for t in e_cc]))).T    
    return e_cc, e_cn, ds_bins
            
def loadFeat(filename):
    loaded = False
    path, file_name = os.path.split(filename)
    cong_level_path = path + "/" + file_name[:-4] + "_level.pkl"
    if os.path.exists("./tcad-data/ViG/" + file_name):
        loaded = True
        return None, None, None, None, None, None, None, filename, loaded
    try:
        with open(filename, "rb") as fin:
            data = pickle.load(fin)
            fin.close()
    except:
        fin = gzip.GzipFile(filename, "rb")
        data = pickle.load(fin)
        fin.close()
    with open(cong_level_path, "rb") as fin:
        level = pickle.load(fin)
        fin.close()
    levelV = level[0]
    levelH = level[1]
    
    density = data['density'][None, :, :]
    rudy = data['rudy'][None, :, :]
    macro = data['macro']
    graph = data['graph']
    layout = data['range']
    macroh, macrov = cal_macro_margin(macro, graph, layout, device='cpu')
    macro = macro[None, :, :]
    macroh = macroh[None, :, :]
    macrov = macrov[None, :, :]
    congH = data['congH'][None, :, :]
    congV = data['congV'][None, :, :]
    levelV = levelV[None, :, :]
    levelH = levelH[None, :, :]
    image = np.concatenate((density, rudy, macro, macroh, macrov))
    label = np.concatenate((congH, congV, levelH, levelV))
    image = torch.tensor(image, dtype=torch.float32, device='cpu')
    label = torch.tensor(label, dtype=torch.float32, device='cpu')
    e_ccs = []
    e_cns = []
    e_cc, e_nn, w_nn, v_n, e_cn, bins = gen_vig_data(data)
    e_ccs.append(e_cc)
    e_cns.append(e_cn)
    e_cc, e_cn, bins = downsample(e_cc, e_cn, bins, 2)
    e_ccs.append(e_cc)
    e_cns.append(e_cn)
    e_cc, e_cn, bins = downsample(e_cc, e_cn, bins, 2)
    e_ccs.append(e_cc)
    e_cns.append(e_cn)
    e_cc, e_cn, bins = downsample(e_cc, e_cn, bins, 2)
    e_ccs.append(e_cc)
    e_cns.append(e_cn)
    return image, label, e_ccs, e_nn, w_nn, v_n, e_cns, filename, loaded

def graph_crop(e_cc, e_cn, bins, meshX, meshY, canvas):
    net = e_cn[1]
    e_cc = e_cc.T
    e_cc = e_cc.tolist()
    e_cc = list(filter(lambda x: \
        x[0]//bins[1] >= canvas[0] and x[0]//bins[1] < canvas[2] and x[0]%bins[1] >= canvas[1] and x[0]%bins[1] < canvas[3] and \
        x[1]//bins[1] >= canvas[0] and x[1]//bins[1] < canvas[2] and x[1]%bins[1] >= canvas[1] and x[1]%bins[1] < canvas[3], e_cc))
    e_cc = np.array(e_cc).T
    e_cc = (e_cc // bins[1] - canvas[0]) * meshY + e_cc % bins[1] - canvas[1]
    e_cn = e_cn.T
    e_cn = e_cn.tolist()
    e_cn = list(filter(lambda x: \
        x[0]//bins[1] >= canvas[0] and x[0]//bins[1] < canvas[2] and x[0]%bins[1] >= canvas[1] and x[0]%bins[1] < canvas[3], e_cn))
    e_cn = np.array(e_cn).T
    if e_cn.shape[0] != 0:
        e_cn[0] = (e_cn[0] // bins[1] - canvas[0]) * meshY + e_cn[0] % bins[1] - canvas[1]
        drop_net_list = list(set(net).difference(e_cn[1]))
    else:
        drop_net_list = []
    return e_cc, e_cn, drop_net_list

def nn_crop(e_nn, e_cn):
    e_nn = e_nn.T
    e_nn = e_nn.tolist()
    e_cn = e_cn.tolist()
    if len(e_cn) == 0:
        e_nn = [[],[]]
        e_nn = np.array(e_nn).T
    else:
        e_nn = list(filter(lambda x: x[0] in e_cn[1] and x[1] in e_cn[1], e_nn))
        e_nn = np.array(e_nn).T
    return e_nn



