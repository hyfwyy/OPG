from matplotlib.pyplot import text
from sklearn.cluster import k_means
from muge_data.data import TextField,DataField
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from muge_data import DataLoader 
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
import time
import warnings
from models3 import MugeModel,MappingType,MugeModelPrefix,Language_model
from transformers import BertConfig, BertForMaskedLM
from muge_data import compute_scores
def save_config(args):
    config_path = os.path.join(args.output_dir,"config")
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    configfile = vars(args)
    config_file_name = time.strftime("%Y-%m-%d=%H:%M:%S", time.localtime()) + ".yaml"
    with open(os.path.join(config_path, config_file_name), "w") as file:
        file.write(yaml.dump(configfile))
def train(dataloader,model,optimizer,scheduler,criterion,args,epoch,lan_model=None,print_freq = 1000):
    model.train()
    batch_time = AverageMeter('Batch time', ':6.3f')
    data_time = AverageMeter('Data time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr = AverageMeter('lr',':.6f')
    output_dir = args.output_dir 
    device = args.device
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    progress = ProgressMeter(len(dataloader),
                             [batch_time, data_time,lr,losses])
    end = time.time()
    for idx, (image_id, image, tokens, all_captions) in enumerate(dataloader):
        image, tokens = image.to(device),tokens.to(device) 
        data_time.update(time.time() - end)
        
        if args.use_aux_loss:
            outputs, aux_loss = model(image, tokens)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            loss_cro = criterion(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
            loss = loss_cro + args.lamda*aux_loss
        else:
            outputs = model(image, tokens)
            logits = outputs.logits[:, args.prefix_length - 1: -1]
            # label = tokens[:,1:]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
        if args.use_kd:
            with torch.no_grad():
                bert_out = lan_model(tokens)
            bert_logits = bert_out.logits[:,1:,:]
            kl_loss = nnf.kl_div(logits[:,:-1,:],bert_logits,'mean')
            loss = loss+ args.theta*kl_loss
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
        # if idx == 1:
        #     break    
def eval(dataloader,model,config,args,epoch=0,val=False,tokenizer=None):
    batch_time = AverageMeter('Batch time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(dataloader), [batch_time, losses],
                             prefix='Evaluation: ')
    print_freq = 1000/args.val_batch_size_xs
    
    model.eval()
    device = args.device
    output = {}
    gt = {}
    model=model.to(device)
    with torch.no_grad():
        for idx,(image_id, image, tokens, all_captions) in enumerate(dataloader):
            # if idx == 10:
            #     break
            end = time.time()
            image = image.to(args.device)
            prefix = model.swin(image)
            prefix_embed = model.clip_project(prefix)
            if val: # greedy for eval, beamseach for test
                result = model.generate_eval_muge(embed = prefix_embed)
            else:
                result = model.generate_test_muge(embed = prefix_embed,beam_size=args.beam_size,generate_prefix=args.generate_prefix)
            # result_list = generate_sin(model,use_beam,args=args,embed=prefix_embed,stop_token_index=stop_token_index)
            # result = result[0][1:-2]
            # if tokenizer is not None:
            #     result = tokenizer.convert_tokens_to_ids(result)
                # result = " ".join('%s' %id for id in result)
            for k,v,g in zip(image_id,result,all_captions):
                v = v[:-2]
                output[k] = " ".join(v).replace("##", "")
                gt[k] = g
            # for k,g in zip(image_id,all_captions):
            #     # output[k] = " ".join(v).replace("##", "")
            #     output[k] = result
            #     gt[k] = g
            # output.append({'image_id':img_id.item(),'caption':result})
            batch_time.update(time.time() - end)
            end = time.time()
      
            if idx % print_freq == 0:
                progress.display(idx)

    cider = compute_scores(gt,output)
    print(cider,flush = True)
    # log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
    # print(log_stats, flush=True)
    if val:
        path_name = args.output_dir.split('/')[-1]
        with open(f'output/{path_name}_log.txt', "a") as f:
            f.write(json.dumps(cider) +f'epoch:{epoch}'+ "\n")
    else:
        if args.generate_prefix is False:
            path_name = args.output_dir.split('/')[-1]
            with open(f'output/{path_name}_log.txt', "a") as f:
                f.write(f'beam:{args.beam_size} '+'test:'+json.dumps(cider) +f'epoch:{epoch}'+ "\n")
        else:
            path_name = args.output_dir.split('/')[-1]
            with open(f'output/{path_name}_log.txt', "a") as f:
                f.write(f'test beam_size:{args.beam_size}'+json.dumps(cider) + "\n")
    return output,cider 
def main(args,config):
    batch_size = args.batch_size_xs
    val_batch_size = args.val_batch_size_xs
    start_epoch = 0
    pre_best_score = 0.0
    textfield = TextField(args.muge_vocab_path, args.muge_dict_path)
    datafield = DataField(args.muge_dataset_path, args.muge_dataset_path, textfield,config=config)
    train_dataset, test_dataset, val_dataset = datafield.splits()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,shuffle=True,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, drop_last=False,shuffle=False,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, drop_last=False,shuffle=False,num_workers=4)
    # load pretrained language model
    bertconfig  = BertConfig.from_json_file(args.bert_config_path)
    mlm_model = BertForMaskedLM.from_pretrained('bert-base-cased', config=bertconfig)
    lan_model = Language_model(mlm_model, bertconfig, args.muge_word_size).to(args.device)
    data = torch.load(args.bert_path)
    lan_model.load_state_dict(data['state_dict'], strict=False)
    # # load model
    # mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = MugeModelPrefix(vocab_path=args.muge_vocab_path,language_model=None, use_kd = args.use_kd,use_aux_loss=args.use_aux_loss, dict_path=args.muge_dict_path,vocab_size=args.muge_word_size)
        print("Train only prefix")
    else:
        model = MugeModel(vocab_path=args.muge_vocab_path,language_model=None,use_kd = args.use_kd,use_aux_loss=args.use_aux_loss,dict_path=args.muge_dict_path,vocab_size=args.muge_word_size)
        print("Train both prefix and GPT")
    sys.stdout.flush()
    if args.eval is False:
        optimizer = AdamW(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))
        total_steps = (args.epochs-start_epoch) * len(train_loader)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.01*total_steps, num_training_steps=total_steps
        )

    # nn的crossloss 是一个类，nn.functional的crossloss是一个函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model = model.to(args.device)
    # training and validation
    if args.eval is False:
        save_config(args)
        if args.checkpoint != '':
            ckpt = torch.load(args.checkpoint,map_location='cpu')
            model.load_state_dict(ckpt['model'])
            # optimizer.load_state_dict(ckpt['optimizer'])
            # scheduler.load_state_dict(ckpt['scheduler'])
            # start_epoch = ckpt['epoch']+1
        print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        for epoch in range(start_epoch,args.epochs): 
            print(f">>> Training epoch {epoch}")
            sys.stdout.flush()
            train(dataloader=train_loader,
                model=model,
                optimizer = optimizer,
                scheduler = scheduler,
                criterion = criterion,
                args=args,
                epoch=epoch,
                lan_model=lan_model
                )
            
            _,cider = eval(dataloader=val_loader,
                                model = model,
                                args=args,
                                config= config,
                                epoch = epoch,
                                val=True
                                )
            recent_score = cider
            pre_best_score = max(recent_score,pre_best_score)
            if pre_best_score <= recent_score:
                torch.save({'epoch':epoch,
                            'model':model.state_dict()},
                        os.path.join(args.output_dir, f"{args.mapping_type}-{epoch:03d}.pt"))
    # test
    else:
        ckpt = torch.load(args.checkpoint,map_location='cpu')
        if 'model' in ckpt.keys():
            ckpt = ckpt['model']
        model.load_state_dict(ckpt,strict=False)
        result,_ = eval(dataloader=val_loader,
                                model = model,
                                args=args,
                                config= config,
                                val=True,
                                tokenizer=textfield.tokenizer
                                )
        if args.save_results:
            path = args.checkpoint.split('/')[-2]
            json.dump(result, open(f'output/{path}_test_results.json', 'w'), indent=4)
