import warnings
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from .mask_tokens import affinity_mask,near_refine,calculate_Attention
# from mmseg.structures import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead_clips_flow
from mmseg.models.utils import *
from .cffm_module.cffm_transformer import BasicLayer3d3   
import torch.nn.functional as F
# from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def create_with_histogram(B,H,W,mask,histogram):
    
    # 将histogram展平，并找到值为0的位置
    # histogram_flattened = histogram.view(B, -1)
    zero_mask_indices = (histogram == 0)

    # 在掩码为0的位置，依据histogram进一步选择
    mask[zero_mask_indices.unsqueeze(1).expand(-1, H * W, -1) & (mask == 0)] = -10000
    return mask

@HEADS.register_module()
class CFFMHead_clips_resize1_8(BaseDecodeHead_clips_flow):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(CFFMHead_clips_resize1_8, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.linear_pred2 = nn.Conv2d(embedding_dim*2, self.num_classes, kernel_size=1)

        depths = decoder_params['depths']
        # if self.training:
        #     input_resolution = (56,56)
        # else:
        #     input_resolution = (55,80)  # dsec 
        input_resolution = (55,80)
        # input_resolution = (25,43)  # ddd17



        self.refine_module = near_refine()
        # self.get_mha = calculate_Attention()
        # """
        self.decoder_focal=BasicLayer3d3(dim=embedding_dim,
               input_resolution = input_resolution,
               depth=depths,
               num_heads=8,
               window_size=7,       # 改window size只需要改这个就好。ddd17:7 dsec:7
               mlp_ratio=4.,
               qkv_bias=True, 
               qk_scale=None,
               drop=0., 
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm, 
               pool_method='fc',
               downsample=None,
               focal_level=2,   # 每个stage两次计算注意力
               focal_window=5, 
               expand_size=3, 
               expand_layer="all",                           
               use_conv_embed=False,
               use_shift=False, 
               use_pre_norm=False, 
               use_checkpoint=False, 
               use_layerscale=False, 
               layerscale_value=1e-4,
               focal_l_clips=[1,1,1],       # 只7*7的window 现在这两个列表没用了，只管list长度是3就行。
               focal_kernel_clips=[7,7,7]

               )

        self.maxpooling = torch.nn.MaxPool2d(kernel_size=8, stride=8)
        self.histogram_norm = nn.LayerNorm(embedding_dim)
        self.mask_size = 7          # mask的局部注意力窗口大小
        # if self.training:
        #     self.pre_mask = self.create_attention_masks(56,56,self.mask_size)
        # else: 
        #     self.pre_mask = self.create_attention_masks(55,80,self.mask_size)
        self.pre_mask = self.create_attention_masks(55,80,self.mask_size)   # DSEC
        # self.pre_mask = self.create_attention_masks(25,43,self.mask_size)   # DDD17
            
        # self.pre_mask = FakeTensor(self.pre_mask)
        
        
    def create_attention_masks(self, H, W, mask_size):
        # 预先分配一个张量来存储所有的mask
        masks = -10000 * torch.ones(H * W, H * W)

        # 遍历每个查询位置
        for q_x in range(H):
            for q_y in range(W):
                half_mask = mask_size // 2
                x_start = max(0, q_x - half_mask)
                x_end = min(H, q_x + half_mask + 1)
                y_start = max(0, q_y - half_mask)
                y_end = min(W, q_y + half_mask + 1)

                # 计算mask矩阵
                temp_mask = -10000 * torch.ones(H, W)
                temp_mask[x_start:x_end, y_start:y_end] = 0

                # 将temp_mask展平并赋值给masks的对应行
                masks[q_x * W + q_y, :] = temp_mask.flatten()
        # print(H, W, "pre mask ready")

        masks = masks.to('cuda')

        return masks

    def forward(self, inputs, batch_size=None, num_clips=None, imgs=None):
        if self.training:
            assert self.num_clips==num_clips            # 4
        # x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32    注释掉，因为输入是c1 2 3 4 的列表
        
        c1, c2, c3, c4, r2_histogram, r3_histogram, t_histogram = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape       # n = batchsize * num_clips, use the torch.stack

        # MLP 统一channel维度
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)   # 1/32

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])

        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)   # 1/16


        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)   # 1/8

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])    # 1/4

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _, _, h, w=_c.shape # 1/4 
        x = self.dropout(_c)

        x = self.linear_pred(x)         

        x = x.reshape(batch_size, num_clips, -1, h, w)      # 自己的输入是 list(B,C,H,W)

        if num_clips!=4:
        # if not self.training:
            print('！！！！！！！！！！！！batch size不符合条件，直接返回segformer特征！！！！！！！！！！')
            return x[:,-1]
        # """

        h2=int(h/2)
        w2=int(w/2)     
        _c = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False) 
        # b d c h w
        _c_further=_c.reshape(batch_size, num_clips, -1, h2, w2)

        # nb,_,nh,nw = c2.shape
        q_target = _c_further[:,-1:]    # 2, 1, 256, 60, 60  这个是target的综合特征，用来计算跨帧注意力的
        kv_ref1 = _c_further[:, 0:1, :, :, :]   # 这个是ref的综合特征，用来计算refined ref
        kv_ref2 = _c_further[:, 1:2, :, :, :]
        kv_ref3 = _c_further[:, 2:3, :, :, :]
        feat_all = []
        feat_all.extend([kv_ref1,kv_ref2,kv_ref3,q_target])


        histograms_temp = []
        histograms = []
        histograms_temp.extend([r2_histogram, r3_histogram, t_histogram])

        for histogram in histograms_temp:
            histogram_pooled = self.maxpooling(histogram)
            B11,_,H11,W11 = histogram_pooled.shape
            histogram = histogram_pooled.permute(0, 2, 3, 1)        # 4400个像素里，仍有1282/1076个是0，说明还是很稀疏的。
            histogram_flattened = histogram.view(B11, H11 * W11, 1).permute(0,2,1).squeeze(1)  # B 1 HW -> B HW
            histograms.append(histogram_flattened)   


        refined_features = self.refine_module(feat_all,histograms,self.pre_mask)

        final_out_1layer = self.decoder_focal(refined_features) # cffm transformer block to calculate cross frame attention;


        _c_further2=torch.cat([_c_further[:,-1], final_out_1layer[:,-1]],1)     
        x2 = self.dropout(_c_further2)
        x2 = self.linear_pred2(x2)  

        x2=resize(x2, size=(h,w),mode='bilinear',align_corners=False)
        return x2

