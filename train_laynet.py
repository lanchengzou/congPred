import pickle
import numpy as np
import os
os.environ['DGLBACKEND'] = 'pytorch'
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as tmp
from einops.layers.torch import Rearrange
from LayNet import LayNet
from utils.metrics import call_metrics
from utils.log import get_logger 
import dgl
import dgl.data
import dgl.sparse as dglsp
from dgl.dataloading import GraphDataLoader
from torchvision import transforms
import random
import argparse
import glob
import multiprocessing as mp
import time
import gzip
from dataViG import ViGSet
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LayNet TRAINING')
parser.add_argument('--log', '-l', type=str, default="LayNet.log")
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--resume', '-r', type=bool, default=False)

EPOCH = 300
TRAIN_BATCH = 180
TEST_BATCH = 20
NUM_GRID = 32
PATCH_SIZE = 8

def init_process_group(world_size, rank):
    dist.init_process_group(
        backend="nccl",  # change to 'nccl' for multiple GPUs
        init_method="tcp://127.0.0.1:12345",
        world_size=world_size,
        rank=rank,
    )

def load_feature(file_path):
    (path, filename) = os.path.split(file_path)
    with open(file_path, "rb") as fin:
        try:
            data = pickle.load(fin)
        except:
            fin.close()
            fin = gzip.GzipFile(file_path, "rb")
            data = pickle.load(fin)
        fin.close()
        v_n = data['v_n']
        w_nn = data['w_nn']
        e_cc = data['e_cc']
        e_cc = e_cc.tolist()
        e_cn = data['e_cn']
        num_gnet = e_cn[1][-1] + 1
        e_cn = e_cn.tolist()
        e_nn = data['e_nn']
        e_nn = e_nn.tolist()
        hg = dgl.heterograph({
                ('Gcell', 'cc', 'Gcell'): (torch.tensor(e_cc[0]), torch.tensor(e_cc[1])),
                ('Gcell', 'cn', 'Gnet'): (torch.tensor(e_cn[0]), torch.tensor(e_cn[1])),
                ('Gnet', 'nc', 'Gcell'): (torch.tensor(e_cn[1]), torch.tensor(e_cn[0])),
                ('Gnet', 'nn', 'Gnet'): (torch.tensor(e_nn[0]), torch.tensor(e_nn[1]))
                },
                {'Gcell': NUM_GRID*NUM_GRID, 'Gnet': num_gnet}
            )
        hg = dgl.add_self_loop(hg, etype='cc')
    with open("./new_collected/" + filename, "rb") as f:
        try:
            data = pickle.load(f)
        except:
            f.close()
            f = gzip.GzipFile("./new_collected/" + filename, "rb")
            data = pickle.load(f)
        v_c = np.concatenate((data['density'][None,:,:],data['rudy'][None,:,:],data['macro'][None,:,:],data['macro_h'][None,:,:],data['macro_v'][None,:,:]))
        v_c = torch.tensor(v_c, dtype=torch.float)
        transforms_size = transforms.Resize(size=(256,256), antialias=True)
        v_c = transforms_size(v_c)
        rearrange_vc = Rearrange('c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = PATCH_SIZE, p2 = PATCH_SIZE)
        v_c = rearrange_vc(v_c)
        hg.ndata['h'] = {'Gcell':v_c, 'Gnet':torch.tensor(v_n, dtype=torch.float)}
        hg.edata['h'] = {'nn':torch.tensor(w_nn)}
        congV = data['congV']
        congH = data['congH']
        congY = np.concatenate((congV[None,:,:],congH[None,:,:])) 
        congY = torch.tensor(congY)
        congY = transforms_size(congY)   
    return (hg, congY)

def load_data(train_ratio=0.9):
    filenames = glob.glob(f"data/ViG/*1.pkl")
    data = {}
    threads = 64
    for idx in range(0, len(filenames), threads):
        begin = time.time()
        pool = mp.Pool(processes=threads)
        procs = []
        for jdx in range(min(threads, len(filenames)-idx)):
            proc = pool.apply_async(load_feature, (filenames[idx+jdx],))
            procs.append(proc)
        pool.close()
        pool.join()
        for jdx, proc in enumerate(procs):
            datum = proc.get()
            data[filenames[idx+jdx]] = datum
        del pool
        del procs
        print(f"\r loaded {idx+threads}/{len(filenames)}, batch time={time.time()-begin}s",end="")
    print("Finish")
    data = list(data.values())
    random.shuffle(data)
    train_num = int(len(filenames)*train_ratio)
    return data[:train_num], data[train_num:]

def adjust_lr_with_warmup(optimizer, step, warm_up_step, dim):
    #lr = dim**(-0.5) * min(step**(-0.5), step*warm_up_step**(-1.5))/100
    lr = min(step**(-0.5), step*warm_up_step**(-1.5))*4e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class PixelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.03, size=256):
        super(PixelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sample_size = size

    def forward(self, features, labels):
        N, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
        labels = labels.reshape(-1)   # (N*H*W,)
        indices = torch.randperm(features.size(0))[:self.sample_size]
        features = features[indices]
        labels = labels[indices]

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # (N*H*W, N*H*W)

        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -mean_log_prob_pos.mean()

        return loss

def train_vig():
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LayNet(chIn=5, chOut=2, chMid=16, img_size=256, patch_size=4, embed_dim=32, 
                 window_size=7, mlp_ratio=2., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 depths=[2, 2, 4, 2], num_heads=[2, 4, 8, 16], out_indices=(0, 1, 2, 3))
    model.to(device)
    print(f"[INFO] USE {torch.cuda.device_count()} GPUs")


    train = ViGSet("trainViG.pkl")
    test = ViGSet("testViG.pkl")
    with open("tcad-exp1-testfiles.pkl", "rb") as ftest:
        testfiles = pickle.load(ftest)
        ftest.close()

    train_loader = GraphDataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = GraphDataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # loss_fn = nn.MSELoss()
    L2_loss = nn.MSELoss()
    CL_loss1 = PixelContrastiveLoss()
    CL_loss2 = PixelContrastiveLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH//4, gamma=0.5)

    start_epoch = -1

    logger_path = "./log/"

    if not os.path.exists(logger_path):
          os.makedirs(logger_path)
          print("-----New Folder -----")
          print("----- " + logger_path + " -----")
          print("----- OK -----")
    save_log_path = logger_path + args.log
    print('GOOOOO!')
    train_loss_recorder = []
    for epoch in range(start_epoch+1, EPOCH):
        train_loss = []
        test_loss = []
        ssim_res = []
        nrms_res = []
        mse2_res = []
        mse5_res = []
        mse10_res = []
        print(f"-------- EPOCH {epoch+1} --------")
        model.train()

        for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):

            batch_x = batch_x.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.float).to(device)
            congPred, y = model(batch_x)

            loss = L2_loss(congPred, batch_y[:,:2,:,:]) + 1e-6*(CL_loss1(y[:,:32,:,:, ], batch_y[:,2,:,:]) + CL_loss2(y[:,32:,:,:, ], batch_y[:,3,:,:]))
            train_loss.append(loss.cpu().data.numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('[TRAIN] Epoch:{}/{}\t loss={:.8f}\t'.format(epoch+1, EPOCH, np.average(train_loss)))
        train_loss_recorder.append(np.average(train_loss))
        scheduler.step()
        model.eval()
        test_recorder_ssim = {}
        test_recorder_nrms = {}
        test_recorder_mse2 = {}
        test_recorder_mse5 = {}
        test_recorder_mse10 = {}
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.to(device)
                batch_y = batch_y[:,:2,:,:].to(device)
                congPred, _ = model(batch_x)

                loss = L2_loss(batch_y, congPred)
                batch_y = batch_y.cpu().data.numpy()
                congPred = congPred.cpu().data.numpy()
                for b in range(0, congPred.shape[0]):
                    ssim, nrms, mse2, mse5, mse10 = call_metrics(batch_y[b], congPred[b])
                    if ssim is None or nrms is None:
                        continue

                    file_name = testfiles[i*BATCH_SIZE+b]
                    design_name, _ = file_name.split("__")
                    if design_name[-5:] == "_flex":
                        design_name = design_name[:-5]
                    if design_name in test_recorder_ssim.keys():
                        test_recorder_ssim[design_name].append(ssim)
                        test_recorder_nrms[design_name].append(nrms)
                        test_recorder_mse2[design_name].append(mse2)
                        test_recorder_mse5[design_name].append(mse5)
                        test_recorder_mse10[design_name].append(mse10)
                    else:
                        test_recorder_ssim[design_name] = [ssim]
                        test_recorder_nrms[design_name] = [nrms]
                        test_recorder_mse2[design_name] = [mse2]
                        test_recorder_mse5[design_name] = [mse5]
                        test_recorder_mse10[design_name] = [mse10]
                test_loss.append(loss.cpu().data.numpy())
        for design_name in test_recorder_ssim.keys():
            ssim_res.append(np.average(test_recorder_ssim[design_name]))
            nrms_res.append(np.average(test_recorder_nrms[design_name]))
            mse2_res.append(np.average(test_recorder_mse2[design_name]))
            mse5_res.append(np.average(test_recorder_mse5[design_name]))
            mse10_res.append(np.average(test_recorder_mse10[design_name]))
        print('[TEST]  Epoch:{}/{}\t loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}\t MSE2={:.6f}\t MSE5={:.6f}\t MSE10={:.6f}'.format(epoch+1, EPOCH, np.average(test_loss),np.average(ssim_res),np.average(nrms_res), np.average(mse2_res),np.average(mse5_res),np.average(mse10_res)))      
        for design_name in test_recorder_ssim.keys():
                print('*****Current Results*****')
                print('[TEST]  Epoch:{}/{}\t loss={:.8f}\t SSIM={:.6f}\t NRMS={:.6f}\t MSE2={:.6f}\t MSE5={:.6f}\t MSE10={:.6f}'.format(epoch+1, EPOCH, np.average(test_loss),np.average(ssim_res),np.average(nrms_res), np.average(mse2_res),np.average(mse5_res),np.average(mse10_res)))      
        

    with open(logger_path + args.log + ".pkl", "wb") as fout:
        pickle.dump(train_loss_recorder, fout)
        fout.close()
