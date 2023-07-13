import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from transformers import BertGenerationTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import utils
from utils import ScstRewardCriterion
import math
import random
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from utils import coco_caption_eval,AverageMeter,ProgressMeter
import yaml
from dataset import create_dataset

import time
import warnings
from models import ClipCaptionModel,MappingType,ClipCaptionPrefix
import os
import datetime
from scheduler import create_scheduler
from optim import create_optimizer
from train_xs import main as main_xs

def scst_train_iter(image, captions_gt, model, scst_criterion, config):
    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module

    if config['sc_baseline_type'] == 'greedy':
        model.eval()
        with torch.no_grad():
            greedy_res = model_without_ddp.generate(image, sample=False, num_beams=1,
                                       max_length=config['max_length'],
                                       min_length=config['min_length'], greedy=True)

    else:
        greedy_res = None

    model.train()
    sample_res, sample_logprobs = model_without_ddp.generate(image, sample=True, num_beams=1, num_return_sequences=config['sc_train_sample_n'],
                                            max_length=config['max_length'], min_length=config['min_length'])

    assert sample_logprobs.requires_grad == True
    # assert sample_res.requires_grad == False

    loss = scst_criterion(captions_gt, greedy_res, sample_res, sample_logprobs)
    return loss


def train_scst(model, data_loader, optimizer, epoch, device, scheduler, scst_criterion, config, global_step=None):
    model.train()

    batch_time = AverageMeter('Batch time', ':6.3f')
    data_time = AverageMeter('Data time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr = AverageMeter('lr',':.7f')
    print(f">>> Training epoch {epoch}",flush=True)
    output_dir = args.output_dir 
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    progress = ProgressMeter(len(data_loader),
                             [batch_time, data_time,lr,losses])
    end = time.time()
    
    print_freq = 1000

    for idx, (image, captions_gt) in enumerate(data_loader):
        image = image.to(device, non_blocking=True)

        loss = scst_train_iter(image, captions_gt, model, scst_criterion, config)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.update(loss.item())
        lr.update(optimizer.param_groups[0]["lr"])
        # torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % print_freq == 0:
            progress.display(idx)
        # if training process broken
        if idx % 5000 == 0:
            torch.save({'model':model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),
                        'epoch':epoch
                        },
                       os.path.join(output_dir, f"model_latest.pt"))

        # global_step += 1
        # if (global_step % config['eval_steps'] == 0) or (global_step >= config['num_training_steps']):
        #     break

    # return global_step


@torch.no_grad()
def evaluation_scst(model, data_loader, device, config):
    # test
    model.eval()

    model_without_ddp = model
    if hasattr(model, 'module'):
        model_without_ddp = model.module
    
    batch_time = AverageMeter('Batch time', ':6.3f')
    progress = ProgressMeter(len(data_loader), [batch_time],
                             prefix='Evaluation: ')
   
    print_freq = 100
    
    result = []

    for idx,(image, image_id) in enumerate (data_loader):
        end = time.time()
        image = image.to(device, non_blocking=True)
        captions = model_without_ddp.generate(image, sample=False, num_beams=config['num_beams'], max_length=config['max_length'],
                                  min_length=config['min_length'])

        for caption, img_id in zip(captions, image_id):
            result.append({"image_id": img_id.item(), "caption": caption})
        
        if idx % print_freq == 0:
            progress.display(idx)
            batch_time.update(time.time() - end)

    return result


def main_scst(args,config):
    
    if args.batch_size > 0:
        config['batch_size_train'] = args.batch_size
    train_dataset, val_dataset, test_dataset = create_dataset('coco_scst', config)
    train_dataset_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], drop_last=True,shuffle=True,num_workers=0,pin_memory=True,collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size_val'], drop_last=False,shuffle=False,num_workers=2,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], drop_last=False,shuffle=False,num_workers=4,pin_memory=True)
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer,'conv':MappingType.CONV}[args.mapping_type]
    # load model
    if args.mapping_type == 'transformer':
        prefix_length = args.prefix_length_tr
        clip_length = args.prefix_length_clip_tr
    else:
        prefix_length = args.prefix_length
        clip_length = args.prefix_length_clip
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length=prefix_length,clip_length=clip_length,num_layers=args.num_layers, 
                                  mapping_type=mapping_type,use_aux_loss=args.use_aux_loss,
                                  return_intermediate=args.return_intermediate,load_swin_ckpt=args.load_swin_ckpt,
                                  batch_size=config['batch_size_train'])
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length=prefix_length, clip_length=clip_length, num_layers=args.num_layers, 
                                 mapping_type=mapping_type,use_sparce_mask=args.use_sparce_mask,load_swin_ckpt=args.load_swin_ckpt,
                                 batch_size=config['batch_size_train'])
        print("Train both prefix and GPT")
    sys.stdout.flush()
    ckpt = torch.load(args.checkpoint,map_location='cpu')
    model.load_state_dict(ckpt,strict=False)
    
    model = model.to(args.device)
    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)
    if args.eval:
        print("Start evaluating",flush=True)
        test_result = evaluation_scst(model, test_loader,args.device, config)
        coco_test = coco_caption_eval(config['test_gt_file'], test_result)
        log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
        print(log_stats, flush=True)
    else:
        print("Start SCST training", flush=True)
        scst_criterion = ScstRewardCriterion(
            cider_cached_tokens=config['cider_cached_tokens'],
            baseline_type=config['sc_baseline_type'])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] ))
        config['num_training_steps'] = arg_sche.num_training_steps
        lr_scheduler = create_scheduler(arg_sche, optimizer)
        # XVLM
        # global_step = 0
        # step_per_epoch = math.ceil(train_dataset_size / (config['batch_size_train']))
        # while global_step < config['num_training_steps']:
        #     epoch = global_step // step_per_epoch
        #     global_step = train_scst(model, train_loader, optimizer, epoch, args.device, lr_scheduler, scst_criterion, config, global_step=global_step)
        #     print(f"### epoch: {epoch}, global_step: {global_step}", flush=True)
        #     test_result = evaluation_scst(model, test_loader, args.device, config)
        #     coco_test = coco_caption_eval(config['test_gt_file'], test_result)
        #     log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
        #     print(log_stats, flush=True)
        # Oscar
        pre_best_score = 0.0
        for epoch in range(args.epoch_scst):
            train_scst(model, train_loader, optimizer, epoch, args.device, lr_scheduler, scst_criterion, config)
            eval_res = evaluation_scst(model, val_loader, args.device, config)
            coco_test = coco_caption_eval(config['val_gt_file'], eval_res)
            log_stats = {**{f'{k}': v for k, v in coco_test.eval.items()}}
            print(log_stats, flush=True)
            cider = coco_test.eval['CIDEr']
            recent_score = cider
            pre_best_score = max(recent_score,pre_best_score)
            if pre_best_score <= recent_score:
                torch.save({'epoch':epoch,
                            'model':model.state_dict()},
                        os.path.join(args.output_dir, f"{epoch:03d}.pt"))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str, default='')
    parser.add_argument('--output_dir', default='checkpoints/temp')
    parser.add_argument('--eval',action='store_true',default=False)
    # parser.add_argument('--eval',type=bool, default=False)
    parser.add_argument('--device',type=str, default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--save_results',type=bool, default=False)
    parser.add_argument('--config_scst',type=str, default='configs/Captioning_scst.yaml')
    parser.add_argument('--config_xs',type=str, default='configs/Captioning.yaml',help='configs/Captioning.yaml or configs/nocaps.yaml')
    parser.add_argument('--batch_size',type=int, default=-1)
    parser.add_argument('--batch_size_xs',type=int, default=30)
    parser.add_argument('--beam_size',type=int, default=1)
    parser.add_argument('--val_batch_size_xs',type=int, default=1)
    parser.add_argument('--epochs',type=int, default=10,help='total epochs in cross enrtopy')
    parser.add_argument('--epoch_scst',type=int, default=5,help='total epochs in self-critical sequence learning')
    parser.add_argument('--num_layers',type=int, default=6)
    parser.add_argument('--load_swin_ckpt',type=bool, default=True)
    parser.add_argument('--scst',type=bool, default=False)
    parser.add_argument('--dataset', default='coco', help='dataset type:coco/nocaps/cc/muge')
    parser.add_argument('--generate_prefix', type=bool, default=False, help='whether generate prefix')
    
    
    # extra setting
    parser.add_argument('--use_aux_loss',type=bool,default=False,help='whether use auxiliary loss function')
    parser.add_argument('--use_sparce_mask',type=bool,default=False,help='whether use learnable mask')
    parser.add_argument('--return_intermediate',type=bool,default=False,help='whether output intermediate of transformer')
    parser.add_argument('--lamda',type=float,default=0.1,help='auxiliary loss parameter')
    parser.add_argument('--threshold',type=float,default=0.5,help='prefix mask threshold')
    
    
    # mlp+gpt
    parser.add_argument('--prefix_length', type=int, default=50,help='prefix length after image encoder') # mlp+gpt
    parser.add_argument('--prefix_length_clip', type=int, default=50,help='real prefix length')# mlp+gpt
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')# mlp+gpt
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer/conv')
    
    # transformer
    parser.add_argument('--prefix_length_tr', type=int, default=50,help='prefix length') # transformer
    parser.add_argument('--prefix_length_clip_tr', type=int, default=50,help='image feature length, fixed') # transformer
    # parser.add_argument('--only_prefix', dest='only_prefix', default=True) # transformer
    # parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')


    args = parser.parse_args()  
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    if args.scst:
        config = yaml.load(open(args.config_scst, 'r'), Loader=yaml.Loader)
        main_scst(args,config)
    else:
        config = yaml.load(open(args.config_xs, 'r'), Loader=yaml.Loader)
        main_xs(args,config)
    