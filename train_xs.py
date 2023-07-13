import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from utils import SwinModel,generate_sin,coco_caption_eval,decode,AverageMeter,ProgressMeter
import yaml
from dataset import create_dataset

import time
import warnings
from models import MappingType,ClipCaptionPrefix,ClipCaptionModel

def save_config(args):
    config_path = os.path.join(args.output_dir,"config")
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    configfile = vars(args)
    config_file_name = time.strftime("%Y-%m-%d=%H:%M:%S", time.localtime()) + ".yaml"
    with open(os.path.join(config_path, config_file_name), "w") as file:
        file.write(yaml.dump(configfile))
# training for single GPU
def train_sin(dataloader,model,optimizer,scheduler,criterion,args,epoch,print_freq = 1000):
    batch_time = AverageMeter('Batch time', ':6.3f')
    data_time = AverageMeter('Data time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr = AverageMeter('lr',':.6f')
    output_dir = args.output_dir 
    device = args.device
    model.train()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    progress = ProgressMeter(len(dataloader),
                             [batch_time, data_time,lr,losses])
    end = time.time()
    for idx, (image, tokens,mask, _) in enumerate(dataloader):
        image,tokens,mask = image.to(device),tokens.to(device),mask.to(device)  
        data_time.update(time.time() - end)
        if args.use_aux_loss:
            outputs, aux_loss = model(image, tokens, mask)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            loss_cro = criterion(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
            # loss = loss_cro + aux_loss*(10**-args.num_layers)
            loss = loss_cro + args.lamda*aux_loss
        else:
            outputs = model(image, tokens, mask)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
         
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        losses.update(loss.item())
        lr.update(optimizer.param_groups[0]["lr"])
        
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
        # if idx == 10:
        #     break
def evaluation_sin(dataloader,model,config,args,epoch=0,val=False):
    batch_time = AverageMeter('Batch time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(dataloader), [batch_time, losses],
                             prefix='Evaluation: ')
    print_freq = 1000/args.val_batch_size_xs
    
    model.eval()
    device = args.device
    output = []
    model=model.to(device)
    with torch.no_grad():
        for idx,(image,img_id) in enumerate(dataloader):
            # if idx == 100:
            #     break
            end = time.time()
            image = image.to(args.device)
            prefix = model.swin(image)
            if args.return_intermediate:
                prefix_embed = model.clip_project(prefix)[-1]
            else:
                prefix_embed = model.clip_project(prefix)
            if args.use_sparce_mask:
                prefix_mask = model.get_mask(prefix_embed)
            else:
                prefix_mask = None
            if val: # greedy for eval, beamseach for test
                result = model.generate_eval(embed = prefix_embed,prefix_mask=prefix_mask)
            else:
                result = model.generate_test(embed = prefix_embed,prefix_mask=prefix_mask,beam_size=args.beam_size,generate_prefix=args.generate_prefix)
            # result_list = generate_sin(model,use_beam,args=args,embed=prefix_embed,stop_token_index=stop_token_index)

            for k,v in zip(img_id,result):
                output.append({'image_id':k.item(),'caption':v})
            # output.append({'image_id':img_id.item(),'caption':result})
            batch_time.update(time.time() - end)
            end = time.time()
      
            if idx % print_freq == 0:
                progress.display(idx)
    
    if val:
        coco_test = coco_caption_eval(config['val_gt_file'], output)
    else:  
        coco_test = coco_caption_eval(config['test_gt_file'], output)
    cider = coco_test.eval['CIDEr']
    print(coco_test.eval,flush = True)
    # log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
    # print(log_stats, flush=True)
    if val:
        path_name = args.output_dir.split('/')[-1]
        with open(f'output/{path_name}_log.txt', "a") as f:
            f.write(json.dumps(coco_test.eval) +f'epoch:{epoch}'+ "\n")
    else:
        if args.generate_prefix is False:
            path_name = args.output_dir.split('/')[-1]
            with open(f'output/{path_name}_log.txt', "a") as f:
                f.write(f'beam:{args.beam_size} '+'test:'+json.dumps(coco_test.eval) +f'epoch:{epoch}'+ "\n")
    return output,cider     
        
def main(args,config):
    batch_size = args.batch_size_xs
    val_batch_size = args.val_batch_size_xs
    if args.mapping_type == 'transformer':
        prefix_length = args.prefix_length_tr
        clip_length = args.prefix_length_clip_tr
    else:
        prefix_length = args.prefix_length
        clip_length = args.prefix_length_clip
    start_epoch = 0
    pre_best_score = 0.0

    # load dataset
    if args.dataset == 'nocaps':
        test_dataset = create_dataset('nocaps',config,args)
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, drop_last=False,shuffle=False,num_workers=4,pin_memory=True)
    else:
        train_dataset,val_dataset,test_dataset = create_dataset('coco',config,args) 
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,shuffle=True,num_workers=4,pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, drop_last=False,shuffle=False,num_workers=4,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, drop_last=False,shuffle=False,num_workers=4,pin_memory=True)
        # load dataloader

    # load model
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length, 
                                num_layers=args.num_layers, mapping_type=mapping_type,use_aux_loss=args.use_aux_loss,
                                return_intermediate=args.return_intermediate,load_swin_ckpt=args.load_swin_ckpt
                                ,use_sparce_mask=args.use_sparce_mask,threshold=args.threshold)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length,
                                num_layers=args.num_layers, mapping_type=mapping_type,use_aux_loss=args.use_aux_loss,
                                batch_size=batch_size,load_swin_ckpt=args.load_swin_ckpt,use_sparce_mask=args.use_sparce_mask,
                                threshold=args.threshold)
        print("Train both prefix and GPT")
    sys.stdout.flush()
    if args.eval is False:
        optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=int(config['warmup_steps']), num_training_steps=(args.epochs-start_epoch) * len(train_loader)
        # )
        total_steps = (args.epochs-start_epoch) * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.01*total_steps, num_training_steps=total_steps
        )

    # nn的crossloss 是一个类，nn.functional的crossloss是一个函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # prefix_model = prefix_model.to(args.device)
    model = model.to(args.device)
    # training and validation
    if args.eval is False:
        save_config(args)
        if args.checkpoint != '':
            ckpt = torch.load(args.checkpoint,map_location='cpu')
            if 'model' in ckpt.keys():
                ckpt = ckpt['model']
            model.load_state_dict(ckpt,strict=False)
            # optimizer.load_state_dict(ckpt['optimizer'])
            # scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch']+1
        print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        for epoch in range(start_epoch,args.epochs): 
            print(f">>> Training epoch {epoch}")
            sys.stdout.flush()
            train_sin(dataloader=train_loader,
                model=model,
                optimizer = optimizer,
                scheduler = scheduler,
                criterion = criterion,
                args=args,
                epoch=epoch
                )
            
            _,bleu4 = evaluation_sin(dataloader=val_loader,
                                model = model,
                                args=args,
                                config= config,
                                epoch = epoch,
                                val=True
                                )
            recent_score = bleu4
            pre_best_score = max(recent_score,pre_best_score)
            if pre_best_score <= recent_score:
                torch.save({'epoch':epoch,
                            'model':model.state_dict()},
                        os.path.join(args.output_dir, f"{args.mapping_type}-best.pt"))
    # test
    else:
        ckpt = torch.load(args.checkpoint,map_location='cpu')
        if 'model' in ckpt.keys():
            ckpt = ckpt['model']
        model.load_state_dict(ckpt,strict=False)
        result,_ = evaluation_sin(dataloader=test_loader,
                                model = model,
                                args=args,
                                config= config,
                                val=False
                                )
        if args.save_results:
            path = args.checkpoint.split('/')[-2]
            json.dump(result, open(f'output/{path}_test_results.json', 'w'), indent=4)
