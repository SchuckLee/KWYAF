import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)
from model import mix_transformer
from mmseg.models.decode_heads import segformer_head
from mmseg.models.decode_heads import cffm_head
# from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

def seg_512_b0_encoder():
    return mix_transformer.mit_b0()

def seg_512_b0_decoder():
    norm_cfg = dict(type='BN', requires_grad=True)
    decode_head=dict(
        # type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    return segformer_head.SegFormerHead(**decode_head)


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


   

class EncoderDecoder(nn.Module):

    def __init__(self,
                 neck=None,
                 train_cfg=dict(),
                 test_cfg=dict(mode='whole'),
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        backbone = mix_transformer.mit_b0()
        # backbone = mix_transformer.mit_b1()   # change this to replace MiT backbone 

        self.backbone = backbone

        if neck is not None:
            pass
            
        
        self._init_decode_head()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    # b0的设置
    def _init_decode_head(self,):
        norm_cfg = dict(type='BN', requires_grad=True)
        decode_head = dict(
            in_channels=[32, 64, 160, 256],  #b0
            # in_channels=[64, 128, 320, 512],    # b1 # change this to replace MiT backbone 
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=256,depths = 1),  # change this to replace MiT backbone (256/768)
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            num_clips = 4
            )
        self.decode_head = cffm_head.CFFMHead_clips_resize1_8(**decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def extract_feat(self, img):
        """Extract features from images."""
        batch_size, _, h, w = img[0].shape 
        print(f'img[0].shape :{img[0].shape }')
        ref1 = img[0]
        ref2 = img[1][:, :3, :, :]
        ref3 = img[2][:, :3, :, :]
        target = img[3][:, :3, :, :]

        stacked_imgs = torch.stack((ref1, ref2, ref3, target), dim=1)   # 先stack为(batch_size, num_clips, C, h, w)
        imgs = stacked_imgs.reshape(batch_size*4, -1, h,w)      # 再reshape为(2*4, 3, 480, 480)
        
        features = self.backbone(imgs)  # c1, c2, c3, c4 = features 这是一个list，是四个尺度的特征。在后面append上3个histogram

        for i in range(len(img)):
            if i==0: 
                continue
            features.append(img[i][:, 3, :, :].unsqueeze(1)) 
       
        return features
    
    
    def encode_decode(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        batch_size, _, h, w = img[0].shape      # B,C,480,480
        x = self.extract_feat(img)  
        out = self.decode_head(x,batch_size,4)
        return out
   
    def whole_inference(self, img, rescale):
        """Inference with full image."""
        seg_logit = self.encode_decode(img)    
        if rescale:     # uosample
            seg_logit = resize(
                seg_logit,
                size=img[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return seg_logit

    def forward(self, img, mode='whole', rescale='True'):
        """Inference with slide/whole style.
        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.
        Returns:
            Tensor: The output segmentation map.
        """
        assert mode in ['slide', 'whole']

        
        if mode == 'slide':
            seg_logit = self.slide_inference(img, rescale)
        else:
            seg_logit = self.whole_inference(img, rescale)


        return seg_logit

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred




if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    inp = torch.rand(1, 3, 200, 346).to(device)
    inp2 = torch.rand(1, 4, 200, 346).to(device)        # 第一帧3通道，随后3帧是四通道（dataset里concat了histogram）
    GT = torch.rand(1, 200, 346).to(device)  
    input_list = [inp, inp2, inp2, inp2]

    model = EncoderDecoder().to(device)
    model.eval()

    print('model params：', sum(p.numel() for p in model.parameters()))

    flops = FlopCountAnalysis(model.cpu(), ([i.cpu() for i in input_list],))
    print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    model.to(device) 



    




