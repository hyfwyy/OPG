import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import os
import pickle
import random
import sys
import argparse
import json
from typing import Tuple, Optional, Union
from utils import generate_sin,coco_caption_eval,decode,AverageMeter,ProgressMeter
import yaml
from dataset import create_dataset
import time
import warnings
from models2 import ConvCapModel



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
        outputs = model(tokens, image, mask)
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

def evaluation_sin(dataloader,model,config,args,epoch=0,val=False,use_beam=False):
    batch_time = AverageMeter('Batch time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(dataloader), [batch_time, losses],
                             prefix='Evaluation: ')
    print_freq = 1000
    
    model.eval()
    device = args.device
    imgid_list,caption_list = [],[]
    results_list=[]
    model=model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    stop_token_index = tokenizer.encode('.')[0]
    with torch.no_grad():
        for idx,(image,img_id) in enumerate(dataloader):
            if val:
                if idx == 2000:
                    break
            end = time.time()
            image = image.to(args.device)
            prefix = model.convnext(image).reshape(1,args.prefix_length, -1)
            prefix_embed = model.l1(prefix)
            
            result_list = generate_sin(model,use_beam,args=args,embed=prefix_embed,stop_token_index=stop_token_index)
            if use_beam:
                    # img_id = [img_id]*args.beam_size
                img_id = [img_id for i in range(args.beam_size)]
                for caption, img_id in zip(result_list, img_id):
                    results_list.append({"image_id": img_id.item(), "caption": caption})
                # time6 = time.time()
                # print('save caption spend {}s'.format(time6-time5))
            else:
                # for caption, img_id in zip(predict_caption, img_id):
                imgid_list.append(img_id.item())
                caption_list.append(result_list)
                
            batch_time.update(time.time() - end)
            end = time.time()
      
            if idx % print_freq == 0:
                progress.display(idx)
    
    result = decode(tokenizer,imgid_list,caption_list)  # !!根据beam—size返回一个beamsize大小的列表，每个列表里是字典
    
    if use_beam:
        total = {}
        for i in range(args.beam_size):
            coco_test = coco_caption_eval(config['test_gt_file'], result[i])
            for k,v in coco_test.eval.items():
                total[k] += v
        for k in total.keys():
            total[k] = total[k]/args.beam_size   
        log_stats = {**{f'test_{k}': v for k, v in total.items()}}
        print(log_stats, flush=True) 
    else:
        if val:
            coco_test = coco_caption_eval(config['val_gt_file'], result)
        else:  
            coco_test = coco_caption_eval(config['test_gt_file'], result)
        meteor = coco_test.eval['METEOR']
        print(coco_test.eval,flush = True)
        # log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
        # print(log_stats, flush=True)
        if val:
            path_name = args.output_dir.split('/')[-1]
            with open(f'output/{path_name}_log.txt', "a") as f:
                f.write(json.dumps(coco_test.eval) +f'epoch:{epoch}'+ "\n")
    return result,meteor     
        
def main(args,config):

    start_epoch = 0
    pre_best_score = 0.0

    # load dataset
    train_dataset,val_dataset,test_dataset = create_dataset(args.dataset,config,args)
    
    # load dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,shuffle=True,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=False,shuffle=True,num_workers=2,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, drop_last=False,shuffle=True,num_workers=2,pin_memory=True)
        
    # load model
    model = ConvCapModel(args.prefix_size,args.prefix_length)
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
        if args.checkpoint != '':
            ckpt = torch.load(args.checkpoint,map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
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
        model.load_state_dict(ckpt)
        result,_ = evaluation_sin(dataloader=test_loader,
                                model = model,
                                args=args,
                                config= config,
                                use_beam=False
                                )
        if args.save_results:
            path = args.checkpoint.split('/')[-1]
        json.dump(result, open(f'output/{path}_test_results.json', 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint',type=str, default='checkpoints/clip-mlp+gpt2/model_latest.pt')
    parser.add_argument('--checkpoint',type=str, default='checkpoints/conv+gpt2/model_latest.pt')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--device',type=str, default='cuda:0')
    parser.add_argument('--save_results',type=bool, default=True)
    parser.add_argument('--beam_size',type=int, default=5)
    parser.add_argument('--output_dir', default='checkpoints/conv+gpt2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size',type=int, default=20)
    parser.add_argument('--val_batch_size',type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=8)
    # parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--config',type=str, default='configs/Captioning.yaml')
    parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
    parser.add_argument('--dataset', default='coco', help='dataset type:coco/nocaps/cc')
    
    # conv+gpt
    parser.add_argument('--prefix_length', type=int, default=50) # mlp+gpt
    parser.add_argument('--prefix_size', type=int, default=50)# mlp+gpt

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