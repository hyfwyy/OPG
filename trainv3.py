from builtins import KeyError
from email.policy import default
from numpy import False_
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from transformers import BertGenerationTokenizer, GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import os
import pickle
import random
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from utils import SwinModel,generate_sin,coco_caption_eval,decode,AverageMeter,ProgressMeter
import yaml
from dataset import create_dataset
import time
import warnings
from models2 import *
import clip

def save_config(args):
    config_path = os.path.join(args.output_dir,"config")
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    configfile = vars(args)
    config_file_name = time.strftime("%Y-%m-%d=%H:%M:%S", time.localtime()) + ".yaml"
    with open(os.path.join(config_path, config_file_name), "w") as file:
        file.write(yaml.dump(configfile))
# training for single GPU
def train(dataloader,model,ImageToPrefixModel,optimizer,scheduler,criterion,args,epoch,print_freq = 1000):
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
        with torch.no_grad():
            if args.prefix_type == 'swin':
                prefix = ImageToPrefixModel(image)   
            elif args.prefix_type == 'clip':
                prefix = ImageToPrefixModel.encode_image(image)
        data_time.update(time.time() - end)
        if args.use_aux_loss:
            outputs, aux_loss = model(tokens, prefix, mask)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            loss_cro = criterion(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
            loss = loss_cro + args.lamda*aux_loss
        else:
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
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

def evaluation(dataloader,model,ImageToPrefixModel,config,args,epoch=0,val=False,use_beam=False):
    batch_time = AverageMeter('Batch time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(dataloader), [batch_time, losses],
                             prefix='Evaluation: ')
    print_freq = 1000
    
    model.eval()
    device = args.device
    imgid_list,caption_list = [],[]
    results_list=[[],[],[],[],[]]
    prefix_model = ImageToPrefixModel
    model=model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('data/gpt2')
    stop_token_index = tokenizer.encode('.')[0]
    with torch.no_grad():
        for idx,(image,img_id) in enumerate(dataloader):
            # if val:
            #     if idx == 2000:
            #         break
            end = time.time()
            image = image.to(args.device)
            if args.prefix_type == 'swin':
                prefix = prefix_model(image)   
            elif args.prefix_type == 'clip':
                prefix = prefix_model.encode_image(image)
            if args.return_intermediate:
                prefix_embed = model.clip_project(prefix)[-1].reshape(1,args.prefix_length, -1)
            else:
                prefix_embed = model.clip_project(prefix).reshape(1,args.prefix_length, -1)
            result_list = generate_sin(model,use_beam,args=args,embed=prefix_embed,stop_token_index=stop_token_index)
            if use_beam:
                assert args.beam_size == 2
                imgid_list.append(img_id.item())
                for i in range(args.beam_size):
                    results_list[i].append(result_list[i])
                # imgid_list.append(img_id.item())
                # caption_list.append(result_list[0])
            else:
                # for caption, img_id in zip(predict_caption, img_id):
                imgid_list.append(img_id.item())
                caption_list.append(result_list)
                
            batch_time.update(time.time() - end)
            end = time.time()
      
            if idx % print_freq == 0:
                progress.display(idx)     
    
    if use_beam:
        coco_test = {}
        for i in range(args.beam_size):
            result = decode(tokenizer,imgid_list,results_list[i])
            if val:
                coco_test_i = coco_caption_eval(config['val_gt_file'], result)
            else:
                coco_test_i = coco_caption_eval(config['test_gt_file'], result)    
  
            for k,v in coco_test_i.eval.items():
                try:
                    coco_test[k] += v
                except KeyError:
                    coco_test[k] = v
        for k in coco_test.keys():
            coco_test[k] = coco_test[k]/args.beam_size   
        log_stats = {**{f'test_{k}': v for k, v in coco_test.items()}}
        print(log_stats, flush=True) 
    else:
        result = decode(tokenizer,imgid_list,caption_list)
        if val:
            coco_test = coco_caption_eval(config['val_gt_file'], result)
        else:  
            coco_test = coco_caption_eval(config['test_gt_file'], result)
        bleu4 = coco_test.eval['Bleu_4']
        print(coco_test.eval,flush = True)
        # log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
        # print(log_stats, flush=True)
    if val:
        path_name = args.output_dir.split('/')[-1]
        with open(f'output/{path_name}_log.txt', "a") as f:
            f.write(json.dumps(coco_test.eval) +f'epoch:{epoch}'+ "\n")
    else:
        path_name = args.checkpoint.split('/')[-1]
        if use_beam:
            with open(f'output/{path_name}_log.txt', "a") as f:
                f.write('test:'+"\n"+json.dumps(coco_test)+"\n")
        else:
            with open(f'output/{path_name}_log.txt', "a") as f:
                f.write('test:'+"\n"+json.dumps(coco_test.eval)+"\n")   
    return result,bleu4     
        
def main(args,config):      
    if args.only_prefix is True:
        args.mapping_type == 'transformer'
        args.prefix_length = args.prefix_length_tr
        args.prefix_length_clip = args.prefix_length_clip_tr
    start_epoch = 0
    pre_best_score = 0.0
    if args.prefix_type == 'swin':
        prefix_model = SwinModel(args.device)
        # args.prefix_length_clip = 50
        if args.mapping_type == 'conv':
            args.prefix_length_clip = 20
    else:
        prefix_model,_ = clip.load('ViT-B/32',device= args.device,jit=False)
        args.prefix_dim = 512
    # load dataset 
    if args.dataset == 'nocaps':
        no_config = yaml.load(open('configs/nocaps.yaml', 'r'), Loader=yaml.Loader)
        test_dataset = create_dataset(args.dataset,no_config,args)
        test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, drop_last=False,shuffle=True,num_workers=2,pin_memory=True)

    else:
        train_dataset,val_dataset,test_dataset = create_dataset(args.dataset,config,args)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,shuffle=True,num_workers=4,pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=False,shuffle=False,num_workers=4,pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, drop_last=False,shuffle=False,num_workers=4,pin_memory=True)
        
    # load model
    # swin_model to deal image as prefix
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer,'conv':MappingType.CONV}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length=args.prefix_length,clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                num_layers=args.num_layers, mapping_type=mapping_type,use_aux_loss=args.use_aux_loss,
                                return_intermediate=args.return_intermediate)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length=args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                num_layers=args.num_layers, mapping_type=mapping_type,use_sparce_mask=args.use_sparce_mask,
                                use_aux_loss=args.use_aux_loss)
        print("Train both prefix and GPT")
    sys.stdout.flush()
    model = model.to(args.device)
    # training and validation
    if args.eval is False:
        save_config(args)
        optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        total_steps = (args.epochs-start_epoch) * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(config['warmup_steps']), num_training_steps=total_steps
        )
    
        # nn的crossloss 是一个类，nn.functional的crossloss是一个函数
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        # prefix_model = prefix_model.to(args.device)
        if args.checkpoint != '':
            ckpt = torch.load(args.checkpoint,map_location='cpu')
            model.load_state_dict(ckpt['model'],strict=False)
            # optimizer.load_state_dict(ckpt['optimizer'])
            # scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt['epoch']+1
        # model.train()
        print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        for epoch in range(start_epoch,args.epochs): 
            print(f">>> Training epoch {epoch}")
            sys.stdout.flush()
            # adjust_learning_rate(optimizer, epoch, config)
            train(dataloader=train_loader,
                model=model,
                ImageToPrefixModel = prefix_model,
                optimizer = optimizer,
                scheduler = scheduler,
                criterion = criterion,
                args=args,
                epoch=epoch
                )
            
            
            _,bleu4 = evaluation(dataloader=val_loader,
                                model = model,
                                ImageToPrefixModel=prefix_model,
                                args=args,
                                config= config,
                                epoch = epoch,
                                use_beam=False,
                                val = True
                                )
            recent_score = bleu4
            pre_best_score = max(recent_score,pre_best_score)
            if pre_best_score <= recent_score:
                torch.save({'epoch':epoch,
                            'model':model.state_dict()},
                        os.path.join(args.output_dir, f"{args.mapping_type}-{epoch:03d}.pt"))
    
    # test
    else:
        ckpt = torch.load(args.checkpoint,map_location='cpu')['model']
        model.load_state_dict(ckpt,strict=False)    
        result,_= evaluation(dataloader=test_loader,
                                model = model,
                                ImageToPrefixModel=prefix_model,
                                args=args,
                                config= config,
                                use_beam=False
                                # val=True
                                )
        if args.save_results:
            path = args.checkpoint.split('/')[-1][:-3]
            json.dump(result, open(f'output/{path}_test_results.json', 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str, default='')
    # parser.add_argument('--checkpoint',type=str, default='checkpoints/swinv1-mlp+gpt2-1024*7680/mlp-008.pt')
    # parser.add_argument('--eval',action='store_true',default=False)
    parser.add_argument('--eval',type=bool,default=True)
    
    parser.add_argument('--prefix_type', type=str, default='swin',help='clip/swin') 
    # parser.add_argument('--prefix_type', type=str, default='swin',help='clip/swin',required=True) 
    parser.add_argument('--device',type=str, default='cuda:0')
    parser.add_argument('--save_results',type=bool, default=False)
    parser.add_argument('--beam_size',type=int, default=1)
    parser.add_argument('--output_dir', default='checkpoints/mlp')
    parser.add_argument('--dataset', default='nocaps', help='dataset type:coco/nocaps/cc')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size',type=int, default=40)
    parser.add_argument('--val_batch_size',type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--prefix_dim',type=int,default=1024,help='image feature dim')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--config',type=str, default='configs/Captioning.yaml',help='configs/nocaps.yaml or configs/Captioning.yaml')
    parser.add_argument('--dist',type=bool, default=False)
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
    
    # extra setting
    parser.add_argument('--use_aux_loss',type=bool,default=False,help='whether use auxiliary loss function')
    parser.add_argument('--use_sparce_mask',type=bool,default=False,help='whether use learnable mask')
    parser.add_argument('--lamda',type=float,default=0.1,help='auxiliary loss parameter')
    parser.add_argument('--return_intermediate',type=bool,default=False,help='whether output intermediate of transformer')
    
    # mlp+gpt
    parser.add_argument('--prefix_length', type=int, default=50,help='prefix length after image encoder') # mlp+gpt
    parser.add_argument('--prefix_length_clip', type=int, default=50,help='real prefix length')# mlp+gpt
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')# mlp+gpt
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer/conv')
    # 
    # transformer
    parser.add_argument('--prefix_length_tr', type=int, default=50,help='prefix length') # transformer
    parser.add_argument('--prefix_length_clip_tr', type=int, default=50,help='image feature length, fixed') # transformer
    # parser.add_argument('--only_prefix', dest='only_prefix', default=True) # transformer
    # parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    print(args.output_dir,flush=True)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    args.nprocs = torch.cuda.device_count()

    main(args,config)