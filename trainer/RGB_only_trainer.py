import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, PolynomialLR
import tqdm
import numpy as np
import random
import time
import os.path as osp
import warnings
from torch.utils.data import Dataset, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2]) 
sys.path.append(root_path)
from model.segformer_build import EncoderDecoder
from utils.utils_func import IOStream
from losses.loss_func import TaskLoss
from dst.dsec_dataset import DSECEvent
from dst.ddd17_dataset import DDD17Event_with_histogram,DDD17Events_ori_temporal
from utils.metrics import MetricsSemseg
from PIL import Image
os.environ['TORCH_FAKE_TENSOR_DEBUG'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
####################
"""VERSION_BACKUP"""
####################
proj_home_path = "your_path/KWYAF"
warnings.filterwarnings("ignore")

def get_local_dir(path):
    return os.path.join(proj_home_path, path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def CityscapesLABELtoRGB(label_arr):
    color_map = {
        0: [70, 130, 180],
        1: [70, 70, 70],
        2: [190, 153, 153],
        3: [220, 20, 60],
        4: [153, 153, 153],
        5: [128, 64, 128],
        6: [244, 35, 232],
        7: [107, 142, 35],
        8: [0, 0, 142],
        9: [102, 102, 156],
        10: [220, 220, 0]
    }

    rgb_arr = np.zeros((label_arr.shape[0], label_arr.shape[1], 3), dtype=np.uint8)
    for key, value in color_map.items():
        rgb_arr[label_arr == key] = value
    return rgb_arr

import torch

import torch
import collections 

def load_matching_weights_pred(model, checkpoint_path):

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    except Exception as e:
        print(f"❌ Error: Failed to load checkpoint file {checkpoint_path}. Details: {e}")
        return
    
    pretrained_dict = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        elif 'Weights' in checkpoint:
            pretrained_dict = checkpoint['Weights']
        else:
            pretrained_dict = checkpoint
    
    if pretrained_dict is None:
        print("❌ Error: No valid weight dictionary found in the checkpoint.")
        return
        
    print(f'✅ Loaded pretrained checkpoint with {len(pretrained_dict)} keys.')
    
    pretrained_dict_clean = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    pretrained_dict_renamed = {}
    
    RENAME_RULES = {
        'decode_head.refine_low_channel': 'decode_head.refine_module.low_channel',
        'decode_head.refine_attn': 'decode_head.refine_module.attn',
    }
    
    for k, v in pretrained_dict_clean.items():
        new_k = k
        is_renamed = False
        
        for old_prefix, new_prefix in RENAME_RULES.items():
            if k.startswith(old_prefix):
                new_k = k.replace(old_prefix, new_prefix)
                is_renamed = True
                break
                
        pretrained_dict_renamed[new_k] = v
    
    model_dict = model.state_dict()

    matched_weights = {}
    unmatched_in_ckpt = []
    
    for k, v in pretrained_dict_renamed.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                matched_weights[k] = v
            else:
                unmatched_in_ckpt.append(
                    f"{k} (🟠 Shape mismatch: pretrained {tuple(v.shape)} vs model {tuple(model_dict[k].shape)})"
                )
        else:
            unmatched_in_ckpt.append(f"{k} (🟠 Name mismatch)")

    uncovered_in_model = [
        k for k in model_dict.keys() if k not in matched_weights
    ]
            
    if matched_weights:
        model_dict.update(matched_weights)
        model.load_state_dict(model_dict, strict=False)
    
    print("\n" + "="*60)
    print("✨ Weight Loading Report")
    print("="*60)
    
    print(f"🟢 Successfully loaded weights: {len(matched_weights)} / {len(pretrained_dict_renamed)}")
    
    print("-" * 60)
    print(f"🟠 Keys in pretrained weights not loaded into the model ({len(unmatched_in_ckpt)}):")
    if unmatched_in_ckpt:
        for i, name in enumerate(unmatched_in_ckpt):
            if i < 10:
                print(f"   -> {name}")
            else:
                print(f"   ... and {len(unmatched_in_ckpt) - 10} more unloaded keys")
                break
    else:
        print("   -> None. All pretrained weights were successfully loaded or matched.")

    print("-" * 60)
    print(f"🔵 Keys in the model not covered by pretrained weights ({len(uncovered_in_model)}):")
    if uncovered_in_model:
        for i, name in enumerate(uncovered_in_model):
            if i < 10:
                print(f"   -> {name}")
            else:
                print(f"   ... and {len(uncovered_in_model) - 10} more uncovered keys")
                break
    else:
        print("   -> None. All model parameters were successfully covered.")
    print("="*60 + "\n")

    return matched_weights

class Trainer():
    
    def __init__(self, args):

        self.args = args
        self.initTime = self.get_local_time()
        self.save_to_dir = self.get_save_dir()
        self.io = IOStream(self.save_to_dir + '/run.log')
        self.io.cprint(str(self.args))

        """Random seed setting"""
        SEED = self.args.seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        """Training device defining"""
        self.args.cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        if args.cuda:
            self.io.cprint(
                'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(
                    torch.cuda.device_count()) + ' devices')
            torch.cuda.manual_seed(self.args.seed)
            torch.backends.cudnn.deterministic = True
        else:
            self.io.cprint('Using CPU')


        """Dataloader init"""
        dsec_seg_path = 'your_path_to_dsec'       # event
        ddd17_seg_path = 'your_path_to_ddd17'
        
        """
        fixed_duration = True: stack events according to time interval; 
                         False: stack events according to number of events.
        """
        train_dataset = DSECEvent(dsec_dir=dsec_seg_path,delta_t_per_data=200,nr_events_window=100000, mode='train', nr_events_data=4, nr_bins_per_data=3,
                        fixed_duration=True, augmentation=True, random_crop=True)
        test_dataset = DSECEvent(dsec_dir=dsec_seg_path,delta_t_per_data=200,nr_events_window=100000, mode='val', nr_events_data=4, nr_bins_per_data=3,
                        fixed_duration=True, augmentation=False, random_crop=False)

        self.train_loader = DataLoader( train_dataset,      
                        num_workers=self.args.num_workers,
                                       batch_size=self.args.batch_size,
                                       shuffle=self.args.shuffle,
                                       drop_last=self.args.drop_last)
        self.test_loader = DataLoader( test_dataset,
                        num_workers=self.args.num_workers,
                                       batch_size=self.args.test_batch_size,
                                       shuffle=False,
                                       drop_last=False)


        """ model setting"""
        self.model = EncoderDecoder()
        decoder_params = count_parameters(self.model.decode_head)
        backbone_params = count_parameters(self.model.backbone) 
        self.io.cprint(f'backbone params: {backbone_params}')
        self.io.cprint(f'decoder params: {decoder_params}') 

        # official weights from SegFormer
        path_weights = "your_path/segformer.b0.512x512.ade.160k.pth" # dict_keys(['meta', 'state_dict', 'optimizer'])
        # weights from KWYAF 
        kwyaf = 'your_path/BestModel.pth'

        """Loading pretrained weights"""
        model_pretrained = torch.load(path_weights)
        print(model_pretrained['meta'].keys())
        pretrained_weights = model_pretrained["state_dict"]
        newParams = self.model.state_dict().copy()
        for (name, param) in newParams.items():
            if name in pretrained_weights:     
                newParams[name] = pretrained_weights[name]

        self.model.decode_head.linear_pred = nn.Conv2d(256, 11, kernel_size=1)  # dsec
        self.model.decode_head.linear_pred2 = nn.Conv2d(256*2, 11, kernel_size=1)  

        load_matching_weights_pred(self.model, kwyaf)



        """Parallel settings"""
        self.model = nn.DataParallel(self.model.to(self.device))

        outstr = "Let's use " + str(torch.cuda.device_count()) + " GPUs!"
        self.io.cprint(outstr)


        if self.args.train_from_checkpoint:
            model_saved = torch.load(self.args.model_path)
            print(model_saved["best_score"], model_saved["Epoch_num"])
            pretrainParams = model_saved["Weights"]
            newParams = self.model.state_dict().copy()
            for (name, param), (name_pretrain, param_pretrain) in zip(newParams.items(), pretrainParams.items()):
                newParams[name] = pretrainParams[name_pretrain]
            self.model.load_state_dict(newParams)

        if self.args.use_sgd:
            self.io.cprint("Use SGD")
            self.opt = optim.SGD([{'params': self.model.parameters()}], lr=self.args.lr, momentum=self.args.momentum,
                                 weight_decay=1e-4)
        else:
            self.io.cprint("Use AdamW")
            self.opt = optim.AdamW([{'params': self.model.parameters(), 'lr': self.args.lr}], weight_decay=0.02, betas= (0.8, 0.99))  # kwyaf

        if self.args.scheduler == 'cos':
            self.scheduler = CosineAnnealingLR(self.opt, self.args.num_epochs,
                                               eta_min=1e-4)
        elif self.args.scheduler == 'step':
            self.scheduler = StepLR(self.opt, step_size=30, gamma=0.5)
        elif self.args.scheduler == 'poly':
            self.scheduler = PolynomialLR(self.opt, total_iters=80, power=1.0, verbose=True)

        """Metrics and Losses setting"""
        # DSEC
        self.matrics_stat = MetricsSemseg(num_classes = 11, ignore_label = 255,
                                        class_names = ['background', 'building', 'fence', 'person', 'pole', 'road',
                                                       'sidewalk', 'vegetation', 'car', 'wall', 'traffic sign'])
        # DDD17
        # self.matrics_stat = MetricsSemseg(num_classes = 6, ignore_label = 255,
        #                                 class_names = ['background', 'building', 'fence', 'person', 'pole', 'road'])
        
        # DSEC
        self.criterion = TaskLoss(losses=['dice', 'cross_entropy'],
                                  gamma=2.0, num_classes=11, alpha=None, weight=None, ignore_index=255)  # dsec:11; ddd17:6
        # DDD17
        # self.criterion = TaskLoss(losses=['dice', 'cross_entropy'],
        #                           gamma=2.0, num_classes=6, alpha=None, weight=None, ignore_index=255)  # dsec:11; ddd17:6


    """Train (single epoch)"""
    def train_pass(self, epoch):

        for param_group in self.opt.param_groups:
            lr_str = "\nlr is: {}".format(param_group['lr'])
            self.io.cprint(lr_str)
        train_loss = 0.0
        target_loss = 0.0
        count = 0.0
        self.model.train()


        for idx, (image, label) in tqdm.tqdm(enumerate(self.train_loader)):
            img = [tensor.float().to(self.device) for tensor in image]
            batch_size = img[0].size()[0]
            if batch_size == 1:
                label = label.to(self.device)
            else:
                label = label.to(self.device).squeeze()

            """model forwarding"""
            logits_pred = self.model(img, mode="whole", rescale=True)  

            """Losses calculating"""
            loss = self.criterion(logits_pred, label)

            """Back-propagating"""
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            """Statistics"""
            count += batch_size
            train_loss += loss.item() * batch_size
            if batch_size == 1:
                 label_pred = torch.argmax(logits_pred, dim=1)
            else:    
                label_pred = torch.argmax(logits_pred.squeeze(), dim=1)
            
            
            if batch_size == 1:
                label_batchsize1 = label.unsqueeze(1).detach()
            else:
                label_batchsize1 = label.unsqueeze(1).detach()

            self.matrics_stat.update_batch(
                label_pred.unsqueeze(1).detach(),
                label_batchsize1)



        """Statistics print"""
        metrics_summary,iou_per_class = self.matrics_stat.get_metrics_summary()
        outstr = 'Train %d, ' \
                 '\nloss: %.2f, Acc: %.2f, mIoU: %.2f' % (epoch, train_loss * 1.0 / count,
                                                           metrics_summary['acc'],
                                                           metrics_summary['mean_iou'])
        self.io.cprint(outstr)
        self.io.cprint(iou_per_class)


    ####################
    """Eval (single epoch)"""

    ####################
    def eval_pass(self, epoch, best_score):

        self.model.eval()
        test_loss = 0.0
        count = 0.0
        pred_kd_loss = 0.0
        mid_kd_loss = 0.0
        target_loss = 0.0
        test_pred = []
        test_true = []
        with torch.no_grad():
            for idx, (image, label) in tqdm.tqdm(enumerate(self.test_loader)):
                batch_size = image[0].size()[0]
                if batch_size == 1:
                    label = label.to(self.device) 
                else:
                    label = label.to(self.device).squeeze()
                img = [tensor.float().to(self.device) for tensor in image]

                
                """Inference"""

                """model forwarding"""
                logits_pred = self.model(img, mode="whole", rescale=True)
                
                """Losses calculating"""
                loss = self.criterion(logits_pred, label)

                """Statistics"""
                count += batch_size
                test_loss += loss.item() * batch_size
                
                if batch_size == 1:
                    label_pred = torch.argmax(logits_pred, dim=1)
                else:    
                    label_pred = torch.argmax(logits_pred.squeeze(), dim=1)

               
                if batch_size == 1:
                    label_batchsize1 = label.detach() 
                else:
                    label_batchsize1 = label.unsqueeze(1).detach()
                self.matrics_stat.update_batch(
                    label_pred.unsqueeze(1).detach(),
                    label_batchsize1)

        """Statistics print"""

        metrics_summary,iou_per_class = self.matrics_stat.get_metrics_summary()
        outstr = 'Test %d, ' \
                 '\nloss: %.2f, Acc: %.2f, mIoU: %.2f' % (epoch, test_loss * 1.0 / count,
                                                           metrics_summary['acc'],
                                                           metrics_summary['mean_iou'])
        self.io.cprint(outstr)
        self.io.cprint(iou_per_class)

        if metrics_summary['mean_iou'] >= best_score:
            best_score = metrics_summary['mean_iou']
            self.save_model(epoch=epoch, best_score=best_score)
            str = "Best Performance is: {}".format(best_score)
            self.io.cprint(str)

        return best_score

    def train(self):

        best_score = 0

        for epoch in range(self.args.num_epochs):
            self.matrics_stat.reset()
            best_score = self.eval_pass(epoch, best_score)
            self.train_pass(epoch)
            self.matrics_stat.reset()
            if epoch % 5 ==0:
                best_score = self.eval_pass(epoch, best_score)


            """Learning rate modurating"""
            if self.args.scheduler == 'cos':
                self.scheduler.step()
            elif self.args.scheduler == 'step':
                if self.opt.param_groups[0]['lr'] > 1e-6:
                    self.scheduler.step()
                if self.opt.param_groups[0]['lr'] < 1e-6:
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = 1e-6
            else:
                self.scheduler.step()

    def eval(self):
        self.eval_pass(epoch=0, best_score=None)

    def get_save_dir(self):

        path = osp.join('your_path/log_file/', self.args.exp_name)
        dir_path = osp.join(path, self.initTime)
        if not osp.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def save_model(self, epoch, best_score=None):
        netDict = self.model.state_dict()
        saveTo = self.save_to_dir

        if best_score is None:
            pass
        else:
            torch.save({
                "Weights": netDict,
                "best_score": best_score,
                "Epoch_num": epoch,
            }, osp.join(saveTo, "BestModel.pth"))

    def get_local_time(self):
        return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())



if __name__ == '__main__':

    # -------------------------------------------------------------------------------------------------------------------- #
    """Training settings"""
    # -------------------------------------------------------------------------------------------------------------------- #

    parser = argparse.ArgumentParser(description='KWYAF')
    

    # """Universal settings"""
    parser.add_argument('--exp_name', type=str, default='KWYAF_B0', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    #
    # """Runing modes"""
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')

    
    parser.add_argument('--model_path', type=str, default='your_path/BestModel.pth',
                         metavar='N',help='Pretrained model path')
    parser.add_argument('--train_from_checkpoint', type=bool, default=False,
                        help='train network from check point')
    parser.add_argument('--num_epochs', type=int, default=80, metavar='N',
                        help='number of episode to train ')
    #
    # """Dataset settings"""
    # parser.add_argument('--training_path', type=str, default='', metavar='N',
    #                     help='path to training data')
    # parser.add_argument('--validation_path', type=str, default='', metavar='N',
    #                     help='path to validation data')
    # parser.add_argument('--num_points', type=int, default=2048,
    #                     help='num of points to inference')
    # parser.add_argument('--sensor_size', type=int, default=[180, 240],
    #                     help='sensor size of cameras')
    # parser.add_argument('--voxel_size', type=int, default=[10, 10, 25 * 1e3],
    #                     help='voxelizating sizes')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num_workers')
    # parser.add_argument('--ifbin', type=int, default=False, metavar='ifbin',
    #                     help='If file type is .bin')
    # parser.add_argument('--augmentation', type=bool, default=True,
    #                     help='If augmentation, e.g., shifting, random interval etc)')
    parser.add_argument('--shuffle', type=int, default=True, metavar='shuffle',
                        help='If shuffle')
    parser.add_argument('--drop_last', type=int, default=False, metavar='drop_last',
                        help='If drop_last')
    # dsec 2
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    # dsec 8
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='test_batch_size',
                        help='Size of batch)')
    #
    # """Model settings"""
    # parser.add_argument('--model_t', type=str, default='resnet18', metavar='N',
    #                     choices=['resnet34', 'resnet18'], help='Model to use')
    parser.add_argument('--seed', type=int, default=3407, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--topk_num', type=int, default=20, metavar='N',
    #                     help='Num of nearest neighbors to use')
    # parser.add_argument('--num_classes', type=int, default=101, metavar='N',
    #                     help='Dataset categories')
    # parser.add_argument('--feat_dim', type=int, default=[25, 64, 64, 128], metavar='N',
    #                     help='Dimensions of input feats')
    # parser.add_argument('--ifpretrain', type=bool, default=True,
    #                     help='if pretrained on ImageNet')
    # parser.add_argument('--ifmultibranch', type=bool, default=True,
    #                     help='if multi-branch prediction')
    # parser.add_argument('--ifnokd', type=bool, default=False,
    #                     help='if multi-branch prediction')
    # parser.add_argument('--num_channel', type=int, default=6, metavar='N',
    #                     help='input channel')
    #
    # """Optimization settings"""
    # parser.add_argument('--use_sgd', type=bool, default=True,
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.00006, metavar='LR',    # dsec
    # parser.add_argument('--lr', type=float, default=0.001, metavar='LR',        # ddd17
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
    #                     help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='poly', metavar='N',
                        choices=['cos', 'step', 'poly'],
                        help='Scheduler to use, [cos, step, poly]')
    #
    args = parser.parse_args()
    #
    trainer = Trainer(args)

    if not args.eval:
        trainer.train()
    # else:
    #     trainer.eval()
