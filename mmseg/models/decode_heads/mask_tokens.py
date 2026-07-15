import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch._subclasses.fake_tensor import FakeTensorMode

def calculate_corr(feat1,feat2):

        batch,_, dim1, ht, wd = feat2.shape
        feat_1 = feat1.reshape(batch*1,dim1,ht,wd)
        feat_1 = feat_1.view(batch*1,dim1,ht*wd)    # B C N
        feat_1 = feat_1.permute(0,2,1)     # B N C
        feat_2 = feat2.reshape(batch*1,dim1,ht,wd)
        feat_2 = feat_2.view(batch*1,dim1,ht*wd)    # B C N
        feat_2 = feat_2.permute(0, 2, 1)     # b n c


        corr = torch.matmul(feat_2, feat_1.transpose(1,2))
        corr = corr.view(batch, ht, wd, 1, ht, wd)  # torch.Size([2, 60, 60, 1, 60, 60])
        corr = corr  / torch.sqrt(torch.tensor(dim1).float())    # torch.Size([2, 60, 60, 1, 60, 60])
        corr = corr.view(batch, ht*wd, ht*wd)

        return corr  
def calculate_corr_histogram(feat1,feat2,f2_histogram):

        batch,_, dim1, ht, wd = feat2.shape
        feat_1 = feat1.reshape(batch*1,dim1,ht,wd)
        feat_1 = feat_1.view(batch*1,dim1,ht*wd)    # B C N
        feat_1 = feat_1.permute(0,2,1)     # B N C
        feat_2 = feat2.reshape(batch*1,dim1,ht,wd)
        feat_2 = feat_2.view(batch*1,dim1,ht*wd)    # B C N
        feat_2 = feat_2.permute(0, 2, 1)     # b n c
        # print('feat2:',feat_2.detach().cpu())
        # print('f2_histogram:',f2_histogram.detach().cpu())
        feat_2 = feat_2 * f2_histogram
        corr = torch.matmul(feat_2, feat_1.transpose(1,2))
        corr = corr.view(batch, ht, wd, 1, ht, wd)  # torch.Size([2, 60, 60, 1, 60, 60])
        corr = corr  / torch.sqrt(torch.tensor(dim1).float())    # torch.Size([2, 60, 60, 1, 60, 60])
        corr = corr.view(batch, ht*wd, ht*wd)

        return corr

class refine_attention(nn.Module):
    def __init__(self, num_classes=11,num_clips=4,embedding_dim=256):
        super(refine_attention, self).__init__()
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.embedding_dim = embedding_dim
        self.v_linear = nn.Linear(embedding_dim,embedding_dim)
    def forward(self, corr,feat,histogram,mask):
        b,n,c = feat.shape
        histogram_flat = histogram.view(b, n)
        histogram_mask = histogram_flat.unsqueeze(2).expand(b, n, n)

        clone_mask = mask.clone()
        
        mask2 = clone_mask.unsqueeze(0).expand(b, -1, -1)
        device = histogram_flat.device

        mask2 = mask2.to(device)

        masked_corr = corr + mask2   # 概率分布Map加上mask，没用的就是-10000，softmax之后接近于0
        corr =  F.softmax(masked_corr, dim=-1)
        ref_feat_v = self.v_linear(feat)
        attention = torch.matmul(corr,ref_feat_v)
        return attention

class near_refine(nn.Module):
    def __init__(self, num_classes=11,num_clips=4,embedding_dim=256):
        super(near_refine, self).__init__()
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.low_channel = nn.Linear(2*embedding_dim,embedding_dim)
        self.attn = refine_attention()
    def forward(self, feats_all,histograms,mask):
        # feats_all: B,1,C,H,W
        # histograms: B,1,H,W
        batch,_, dim, ht, wd = feats_all[0].shape

        feats_reshaped = []
        for feat_1 in feats_all:
            feats_reshaped.append(feat_1)

        final_feats = []
        final_feats.append(feats_reshaped[0])   # 1 加I1

        corr_r2 = calculate_corr(feat1=feats_reshaped[0],feat2=feats_reshaped[1])
        feats_reshaped2 = feats_reshaped[1].reshape(batch*1,dim,ht,wd)
        feats_reshaped2 = (feats_reshaped2.view(batch*1,dim,ht*wd)).permute(0,2,1)
        motion_r2 = self.attn(corr_r2,feats_reshaped2,histograms[0],mask)
        refined_r2 = motion_r2 + feats_reshaped2
        refined_r2_cat =  self.low_channel(torch.cat([refined_r2,feats_reshaped2],dim=-1))
        final_feats.append(refined_r2_cat)      # 1 加refined I2

        corr_r3 = calculate_corr(feat1=refined_r2,feat2=feats_reshaped[2])
      

        feats_reshaped3 = feats_reshaped[2].reshape(batch*1,dim,ht,wd)
        feats_reshaped3 = (feats_reshaped3.view(batch*1,dim,ht*wd)).permute(0,2,1)  # B N C
        motion_r3 = self.attn(corr_r3,feats_reshaped3,histograms[1],mask) 
        refined_r3 = motion_r3 + feats_reshaped3 
        refined_r3_cat = self.low_channel(torch.cat([refined_r3,feats_reshaped3],dim=-1))
        final_feats.append(refined_r3_cat)      # 2 加refined I3

        corr_t = calculate_corr(feat1=refined_r3,feat2=feats_reshaped[3])
      
        feats_reshaped4 = feats_reshaped[3].reshape(batch*1,dim,ht,wd)
        feats_reshaped4 = (feats_reshaped4.view(batch*1,dim,ht*wd)).permute(0,2,1)  # 原来的target特征
        motion_t = self.attn(corr_t,feats_reshaped4,histograms[2],mask) 
        refined_t = motion_t  + feats_reshaped4   # B N C
        refined_t_cat = self.low_channel(torch.cat([refined_t,feats_reshaped4],dim=-1))
        final_feats.append(refined_t_cat)       # 3 加refined I4
        
       
        transformed = []
        
        for i in range(len(final_feats)):
            if i==0:
                transformed.append(final_feats[0])
                continue        # 这几行是r1 r2' r3'和t' 记得改上面的1 2 3 4注意好加什么

            ff = (final_feats[i].permute(0,2,1)).reshape(batch*1,dim,ht,wd)
            ff = ff.unsqueeze(1)        
            transformed.append(ff)
            # ff:  torch.Size([2, 1, 256, 55, 80])
        cat_feats = torch.cat(transformed,dim = 1)
        return cat_feats  

class calculate_Attention(nn.Module):
    def __init__(self, num_classes=11,num_clips=4,embedding_dim = 256):
        super(calculate_Attention, self).__init__()
        self.num_classes = num_classes
        self.num_clips = num_clips
        self.fuse_module = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )
        # self.attention = nn.MultiheadAttention(embed_dim = embedding_dim, num_heads = 8)
        self.q_proj = nn.Linear(embedding_dim,embedding_dim)
        self.k_proj = nn.Linear(embedding_dim,embedding_dim)
        self.v_proj = nn.Linear(embedding_dim,embedding_dim)

    def forward(self,refined_target,f_all):
        batch3, dim3, h3, w3 = refined_target.shape
       
        feats = []
        for _f in f_all:
            _f = _f.reshape(batch3*1,dim3,h3,w3)
            print('_f',_f.shape)
            feats.append(_f)
        concatenated_feats = torch.cat(feats, dim=1)
        # 2, 256*4, 55, 80
        concatenated_feats = self.fuse_module(concatenated_feats)   # 2, 256, 55, 80
        N = h3 * w3
        B = batch3
        C = dim3
        q_refined_target = refined_target.permute(2, 3, 0, 1).reshape(N, B, C)
        kv_cat_feats = concatenated_feats.permute(2, 3, 0, 1).reshape(N, B, C)
        q_projed = self.q_proj(q_refined_target)
        k_projed = self.k_proj(kv_cat_feats)
        v_projed = self.v_proj(kv_cat_feats)
        atten_score = torch.matmul(q_projed,k_projed.transpose(1,2))
        atten_score = atten_score / torch.sqrt(torch.tensor(C).float())
        atten_score = F.softmax(atten_score)
        motion_attn = torch.matmul(atten_score,v_projed)
        motion_attn = (motion_attn.permute(1,2,0)).reshape(batch3*1,dim3,h3,w3)
        print('motion_attn的形状',motion_attn.shape)    # B C H W
        return motion_attn

class ori_near_mask(nn.Module):
    def __init__(self, num_classes=11,num_clips=4):
        super(ori_near_mask, self).__init__()
        self.num_classes = num_classes
        self.num_clips = num_clips

        
    def forward(self, refs,big_target):

        batch,_, dim1, ht, wd = big_target.shape
        correlations = []
        refs.append(big_target)
        feats_reshaped = []
        for feat_1 in refs:
            feat_1 = feat_1.reshape(batch*1,dim1,ht,wd)
            feat_1 = feat_1.view(batch*1,dim1,ht*wd)  # raft方式要注释掉
            feats_reshaped.append(feat_1)

        for idx in range(len(feats_reshaped)-1):
            
            corr = torch.matmul(feats_reshaped[idx].transpose(1,2), feats_reshaped[idx+1])
            corr = corr.view(batch, ht, wd, 1, ht, wd)  # torch.Size([2, 60, 60, 1, 60, 60])
            corr = corr  / torch.sqrt(torch.tensor(dim1).float())    # torch.Size([2, 60, 60, 1, 60, 60])
            batch, h1, w1, dim2, h2, w2 = corr.shape
            corr = corr.reshape(batch*h1*w1, dim2, h2, w2)
            corr = F.avg_pool2d(corr, 2, stride=2)
            _,_,new_h2,new_w2 = corr.shape
            corr = corr.reshape(batch,h1,w1,new_h2*new_w2)
            """
            coords_0,coords1 = initialize_flow(feats_reshaped[idx])
            corr_fn = CorrBlock(feats_reshaped[idx],feats_reshaped[idx+1])
            corr = corr_fn(coords1)
            """
            correlations.append(corr)   
        return correlations     
  
class affinity_mask(nn.Module):
    def __init__(self, num_classes=11,num_clips=4):
        super(affinity_mask, self).__init__()
        self.num_classes = num_classes
        self.num_clips = num_clips        


    def forward(self, refs ,big_target):
      
        batch,_, dim1, ht, wd = big_target.shape
        big_target = big_target.reshape(batch*1,dim1,ht,wd)
       
        correlations = []
        for ref in refs:
            print('出bug的ref:',ref.shape)
            ref = ref.reshape(batch*1,dim1,ht,wd)   # torch.Size([2, 128, 55, 80])
            
            coords_0,coords1 = initialize_flow(ref)
            print(f'coords1.shape{coords1.shape}')
            # ([2, 2, 55, 80])
            print('big_target.shape',big_target.shape)
            corr_fn = CorrBlock(ref,big_target)
            corr = corr_fn(coords1)

            print('corr.shape',corr.shape)  # torch.Size([2, 324, 55, 80])
            correlations.append(corr)   
        return correlations
       
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
def initialize_flow(img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device) 
        coords1 = coords_grid(N, H, W, device=img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4): 
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device) # 单层邻域大小：2*3+1 = 7， 7*7*4 = 49*4=196
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())

