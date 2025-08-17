import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    batch_s = windows.shape[0] 
    fenmu = H * W / window_size / window_size
    B = batch_s // fenmu        # 使用整除规避int()的bug
    B = math.floor(B)   
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention3d3(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        expand_size (int): The expand size at focal level 1.
        window_size (tuple[int]): The height and width of the window.
        focal_window (int): Focal region size.
        focal_level (int): Focal attention level.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0 
        pool_method (str): window pooling method. Default: none
    """

    def __init__(self, dim, expand_size, window_size, focal_window, focal_level, num_heads, 
                    qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none", focal_l_clips=[7,1,2], focal_kernel_clips=[7,5,3]):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.focal_level = focal_level
        self.focal_window = focal_window
        num_clips=4
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) 
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.focal_l_clips=focal_l_clips
        self.focal_kernel_clips=focal_kernel_clips



    def forward(self, x_all, batch_size=None, num_clips=None):
        """
        Args:
            x_all (list[Tensors]): input features at different granularity
            x_all[0][0]是segformer提取的target feat；x_all[0][1]是target pooled；
            x_all[1 2 3]是reference pooled feats
            mask_all (list[Tensors/None]): masks for input features at different granularity
        """
        
        x = x_all[0][0] # 第一个0是target的,第二个是pool的最大的特征;[0][0]是segformer提取的target feat；

        B0, nH, nW, C = x.shape       
        assert B0==batch_size
        qkv = self.qkv(x).reshape(B0, nH, nW, 3, C).permute(3, 0, 1, 2, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B0, nH, nW, C

        (q_windows, k_windows, v_windows) = map(    
            lambda t: window_partition(t, self.window_size[0]).view(
            -1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads
            ).transpose(1, 2), 
            (q, k, v)
        )
        if self.expand_size > 0 and self.focal_level > 0:
            (k_tl, v_tl) = map( 
                lambda t: torch.roll(t, shifts=(-self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_tr, v_tr) = map(
                lambda t: torch.roll(t, shifts=(-self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_bl, v_bl) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size, -self.expand_size), dims=(1, 2)), (k, v)
            )
            (k_br, v_br) = map(
                lambda t: torch.roll(t, shifts=(self.expand_size, self.expand_size), dims=(1, 2)), (k, v)
            )        
            (k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows) = map(
                lambda t: window_partition(t, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads), 
                (k_tl, k_tr, k_bl, k_br)
            )            
            (v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows) = map(
                lambda t: window_partition(t, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads), 
                (v_tl, v_tr, v_bl, v_br)
            )
            k_rolled = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 1).transpose(1, 2)
            v_rolled = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 1).transpose(1, 2)

            k_rolled = torch.cat((k_windows, k_rolled), 2)  
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows; v_rolled = v_windows; 

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            # 对target pooled feats处理
            for k in range(self.focal_level-1):
                stride = 2**k
                x_window_pooled = x_all[0][k+1]  
                nWh, nWw = x_window_pooled.shape[1:3] 
                # generate k and v for pooled windows               
                B7,nh7,nw7,windowsize,_,dim7 = x_window_pooled.shape
                num_heads = 8
                x_window_pooled = x_window_pooled.view(B7*nh7*nw7, windowsize*windowsize, dim7)
                x_window_pooled = x_window_pooled.view(B7 * nh7 * nw7, num_heads, windowsize * windowsize, dim7 // num_heads)

                qkv_pooled = x_window_pooled.unsqueeze(0).repeat(3, 1, 1, 1, 1)  
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]  # B0, C, nWh, nWw
                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]
            # 对ref pooled feats处理
            for k in range(len(self.focal_l_clips)):
                
                x_window_pooled = x_all[k+1]    
                nWh, nWw = x_window_pooled.shape[1:3] 
                

                B7,nh7,nw7,windowsize,_,dim7 = x_window_pooled.shape
                num_heads = 8
                x_window_pooled = x_window_pooled.view(B7*nh7*nw7, windowsize*windowsize, dim7)
                qkv_pooled = self.qkv(x_window_pooled).reshape(B7*nh7*nw7, windowsize*windowsize, 3, C)
                # print('qkv_pooled',qkv_pooled.shape)    # 192, 49, 3, 256
                k_pooled_k, v_pooled_k = qkv_pooled[:,:,1], qkv_pooled[:,:,2]  # B0, C, nWh, nWw
                # print('k_pooled_k',k_pooled_k.shape)
                k_pooled_k = k_pooled_k.view(B7 * nh7 * nw7,  windowsize * windowsize, num_heads, dim7 // num_heads).permute(0,2,1,3)
                v_pooled_k = v_pooled_k.view(B7 * nh7 * nw7,  windowsize * windowsize, num_heads, dim7 // num_heads).permute(0,2,1,3)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2] 

        q_windows = q_windows * self.scale 

        attn = (q_windows @ k_all.transpose(-2, -1))  # B0*nW, nHead, window_size*window_size, focal_window_size*focal_window_size

        window_area = self.window_size[0] * self.window_size[1]       
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area, C)

      
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N, window_size, unfold_size):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N        
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * ((window_size + 2*self.expand_size)**2-window_size**2)          

        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        if self.pool_method != "none" and self.focal_level > 1:
            flops += self.num_heads * N * (self.dim // self.num_heads) * (unfold_size * unfold_size)          
        if self.expand_size > 0 and self.focal_level > 0:
            flops += self.num_heads * N * (self.dim // self.num_heads) * ((window_size + 2*self.expand_size)**2-window_size**2)          

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class CffmTransformerBlock3d3(nn.Module):
    r""" Focal Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        expand_size (int): expand size at first focal level (finest level).
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm 
        pool_method (str): window pooling method. Default: none, options: [none|fc|conv]
        focal_level (int): number of focal levels. Default: 1. 
        focal_window (int): region size of focal attention. Default: 1
        use_layerscale (bool): whether use layer scale for training stability. Default: False
        layerscale_value (float): scaling value for layer scale. Default: 1e-4
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_level=1, focal_window=1, use_layerscale=False, layerscale_value=1e-4, focal_l_clips=[7,2,4], focal_kernel_clips=[7,5,3]):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method  # 'fc'
        self.focal_level = focal_level  # 2
        self.focal_window = focal_window    # 5
        self.use_layerscale = use_layerscale    
        self.focal_l_clips=focal_l_clips    # [1,2,3]
        self.focal_kernel_clips=focal_kernel_clips  # [7,5,3]

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size
        
        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention3d3(
            dim, expand_size=self.expand_size, window_size=to_2tuple(self.window_size), 
            focal_window=focal_window, focal_level=focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, pool_method=pool_method, focal_l_clips=focal_l_clips, focal_kernel_clips=focal_kernel_clips)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
        self.kv_windows = [7,5,3]
        

    def forward(self, x):
        H0, W0 = self.input_resolution
        B0, D0, H0, W0, C = x.shape  
        shortcut = x   
        x=x.reshape(B0*D0,H0,W0,C).reshape(B0*D0,H0*W0,C)
        x = self.norm1(x)
        x = x.reshape(B0*D0, H0, W0, C) # 2*4 = 8, 60, 60, C
     
        pad_l = pad_t = 0
        pad_r = (self.window_size - W0 % self.window_size) % self.window_size   # 7-4=3
        pad_b = (self.window_size - H0 % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        
        B, H, W, C = x.shape    
        print('2.----------------x.shape',x.shape)    # ([8, 63, 63, 256])
        if self.shift_size > 0: 
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        shifted_x=shifted_x.view(B0,D0,H,W,C)
        x_windows_all = [shifted_x[:,-1]]       
        x_windows_all_clips=[]  
       
        if self.focal_level > 1 and self.pool_method != "none": 
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level-1):  
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))   
                x_level_k = shifted_x[:,-1] 
                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(), window_size_glo) 
                # B0, nw, nw, window_size, window_size, C    
                x_windows_all += [x_windows_noreshape]     # Ft
            x_windows_all_clips += [x_windows_all]
            for k in range(len(self.focal_l_clips)): 
                
                window_size_glo = self.window_size_glo
                x_level_k = shifted_x[:,k]  
                x_windows_noreshape = window_partition_noreshape(x_level_k.contiguous(), window_size_glo) # B0, nw, nw, window_size, window_size, C    
                x_windows_all_clips += [x_windows_noreshape]   
        attn_windows = self.attn(x_windows_all_clips, batch_size=B0, num_clips=D0)  # nW*B0, window_size*window_size, C
        attn_windows = attn_windows[:, :self.window_size ** 2]
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H(padded) W(padded) C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x[:, :H0, :W0].contiguous().view(B0, -1, C)
        
        x = shortcut[:,-1].view(B0, -1, C) + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))
       
        x=torch.cat([shortcut[:,:-1],x.view(B0,H0,W0,C).unsqueeze(1)],1)

        assert x.shape==shortcut.shape

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size, self.window_size, self.focal_window)

        if self.pool_method != "none" and self.focal_level > 1:
            for k in range(self.focal_level-1):
                window_size_glo = math.floor(self.window_size_glo / (2 ** k))
                nW_glo = nW * (2**k)
                # (sub)-window pooling
                flops += nW_glo * self.dim * window_size_glo * window_size_glo         
                # qkv for global levels
                # NOTE: in our implementation, we pass the pooled window embedding to qkv embedding layer, 
                # but theoritically, we only need to compute k and v.
                flops += nW_glo * self.dim * 3 * self.dim       

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer3d3(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        expand_size (int): expand size for focal level 1. 
        expand_layer (str): expand layer. Default: all
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm 
        pool_method (str): Window pooling method. Default: none. 
        focal_level (int): Number of focal levels. Default: 1.
        focal_window (int): region size at each focal level. Default: 1. 
        use_conv_embed (bool): whether use overlapped convolutional patch embedding layer. Default: False 
        use_shift (bool): Whether use window shift as in Swin Transformer. Default: False 
        use_pre_norm (bool): Whether use pre-norm before patch embedding projection for stability. Default: False
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False. 
        use_layerscale (bool): Whether use layer scale for stability. Default: False.
        layerscale_value (float): Layerscale value. Default: 1e-4.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer="all",
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_level=1, focal_window=1, use_conv_embed=False, use_shift=False, use_pre_norm=False, 
                 downsample=None, use_checkpoint=False, use_layerscale=False, layerscale_value=1e-4, focal_l_clips=[16,8,2], focal_kernel_clips=[7,5,3]):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth  # 1
        self.use_checkpoint = use_checkpoint

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1
        # build blocks
        self.blocks = nn.ModuleList([
            CffmTransformerBlock3d3(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,  # 7
                                 shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                 expand_size=0 if (i % 2 == expand_factor) else expand_size, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pool_method=pool_method, 
                                 focal_level=focal_level,   # 2
                                 focal_window=focal_window,     # 5
                                 use_layerscale=use_layerscale, 
                                 layerscale_value=layerscale_value,
                                 focal_l_clips=focal_l_clips,   # [1,2,3]
                                 focal_kernel_clips=focal_kernel_clips) # [7,5,3]
            for i in range(depth)]) 

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size=2, in_chans=dim, embed_dim=2*dim, 
                use_conv_embed=use_conv_embed, norm_layer=norm_layer, use_pre_norm=use_pre_norm, 
                is_stem=False
            )
        else:
            self.downsample = None
    def forward(self, x, batch_size=None, num_clips=None):
        B, D, C, H, W = x.shape # torch.Size([2, 4, 256, 60, 60])
        x = rearrange(x, 'b d c h w -> b d h w c')
        for blk in self.blocks:
            if self.use_checkpoint:
                new_x = checkpoint.checkpoint(blk, x)
            else:
                new_x = blk(x)
        _x = new_x
        if self.downsample is not None:
            _x = _x.view(_x.shape[0], self.input_resolution[0], self.input_resolution[1], -1).permute(0, 3, 1, 2).contiguous()
            _x = self.downsample(_x)
        _x = rearrange(_x, 'b d h w c -> b d c h w')
        return _x
        
    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

