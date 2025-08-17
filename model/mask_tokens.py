import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch._subclasses.fake_tensor import FakeTensorMode

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
        # with FakeTensorMode(allow_non_fake_inputs=True):
        #     # 在FakeTensorMode上下文中使用预定义的Tensor
        #     clone_mask = mask.clone()
        clone_mask = mask.clone()
        
        mask2 = clone_mask.unsqueeze(0).expand(b, -1, -1)
        device = histogram_flat.device
        # 取histogram_flat的设备

        mask2 = mask2.to(device)
        
        # 消融实验，去掉mask的最佳window size
        # mask2[histogram_mask == 0] = -10000
        
        masked_corr = corr + mask2   # 概率分布Map加上mask，没用的就是-10000，softmax之后接近于0
        corr =  F.softmax(masked_corr, dim=-1)
        # softmax后，0的位置代表不需要计算的token
        # zero_count = (corr == 0).sum().item()
        # print('3corr中----------0的个数为：',zero_count)  # 22310096是batchsize=2的。除以2除以（4400*4400）=0.576，去除了57.6%的token。
        ref_feat_v = self.v_linear(feat)
        attention = torch.matmul(corr,ref_feat_v)
        return attention

# 这个是直接softmax乘以k的版本
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
        # 消融实验，不用精化的去优化
        # corr_r3 = calculate_corr(feat1=feats_reshaped[1],feat2=feats_reshaped[2])

        feats_reshaped3 = feats_reshaped[2].reshape(batch*1,dim,ht,wd)
        feats_reshaped3 = (feats_reshaped3.view(batch*1,dim,ht*wd)).permute(0,2,1)  # B N C
        # motion_r3 = self.attn(corr_r3,feats_reshaped3) 
        motion_r3 = self.attn(corr_r3,feats_reshaped3,histograms[1],mask) 
        refined_r3 = motion_r3 + feats_reshaped3 
        refined_r3_cat = self.low_channel(torch.cat([refined_r3,feats_reshaped3],dim=-1))
        final_feats.append(refined_r3_cat)      # 2 加refined I3

        corr_t = calculate_corr(feat1=refined_r3,feat2=feats_reshaped[3])
        # 消融实验，不用精化的去优化
        # corr_t = calculate_corr(feat1=feats_reshaped[2],feat2=feats_reshaped[3])

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


