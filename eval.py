from xml.etree.ElementPath import prepare_descendant
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer,GPT2LMHeadModel
import sys
import argparse
import json
from utils import SwinModel,generate_prefix,coco_caption_eval,decode,AverageMeter,ProgressMeter
import yaml
from dataset import create_dataset,create_minibatch
import time
from models2 import *
import clip
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# num: number of image and caption pair
def get_prefix(args,prefix_model,num):
    minibatch = create_minibatch(num)
    batch = len(minibatch)
    if args.prefix_type == 'swin':
        prefix_dim = 1024
    else:
        prefix_dim = 512
    # max_dim = max(768,prefix_dim)
    max_dim = 768
    prefix_all = torch.zeros((batch,50,max_dim))
    caption_all  = torch.zeros((batch,50,max_dim))
    tokens_all = []
    gpt = GPT2LMHeadModel.from_pretrained('gpt2')
    model = ClipCaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                num_layers=args.num_layers, mapping_type=args.mapping_type)
    ckpt = torch.load(args.checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.to(args.device)
    with torch.no_grad():
        for i in range(len(minibatch)):
            item = minibatch[i]
            image = item['image']
            image_id = item['image_id']
            tokens = item['tokens']
            image = image.unsqueeze(0)
            image = image.to(args.device)
            if args.prefix_type == 'swin':
                prefix = prefix_model(image).squeeze(0) 
                prefix_embed = model.clip_project(prefix)
                prefix_all[i,:prefix.shape[0],:prefix.shape[1]] = prefix_embed
            elif args.prefix_type == 'clip':
                prefix = prefix_model.encode_image(image)
                prefix_all[i,:prefix.shape[0],:prefix.shape[1]] = prefix_embed
            tokens_embed = gpt.transformer.wte(tokens)
            caption_all[i,:tokens_embed.shape[0],:tokens_embed.shape[1]] = tokens_embed
            tokens_all.append(tokens) 
    return prefix_all,caption_all,tokens_all


# preix:[B,50,1024] caption[B,30,768]
def minibatch_cos(args):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    if args.prefix_type == 'swin':
        prefix_model = SwinModel(args.device)
    else:
        prefix_model,_ = clip.load('ViT-B/32',device= args.device,jit=False)
        
    prefix,token_embed,token = get_prefix(args,prefix_model,5)
    # prefix = prefix.squeeze(0)
    # token_embed = token_embed.squeeze(0)
    prefix = prefix.reshape(-1,prefix.shape[-1])
    token_embed = token_embed.reshape(-1,token_embed.shape[-1])
    caption = [tokenizer.decode(i) for i in token]
    # img_label = [i for i in range(prefix.shape[0])]
    labels = []
    # stop_token = '.'
    # caption = caption[0][:-1]
    for j in range(len(caption)):
        for i in caption[j].split():
            labels.append(i)
    # labels.append(stop_token)
    token_lenth = [len(token[i]) for i in range(len(token))]
    sum=0    
    for i in token_lenth:
        sum+=i
    token_embed = token_embed[:sum]
    torch.norm(token_embed,dim=0)
    # embed_all = torch.cat((nnf.normalize(token_embed),nnf.normalize(prefix)))
    embed_all = nnf.normalize(token_embed)
    img_label = [i for i in range(embed_all.shape[0]-len(labels))]
    for i in img_label:
        labels.append(i)  # annotaions
    tsne_model = TSNE(perplexity=40,n_components=2,init='pca',n_iter=6000, random_state=23)
    new_values = tsne_model.fit_transform(embed_all)
    x,y=[],[]
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    # plt.figure(figsize=(16, 16)) #定义画布大小
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    plt.savefig('1.jpg')
    # prefix = prefix.reshape(prefix.shape[0],-1)
    # caption = caption.reshape(caption.shape[0],-1)
    # cos_matix = torch.zeros((prefix.shape[0],prefix.shape[0]))
    # for i in range(prefix.shape[0]):
    #     for j in range(prefix.shape[0]):
    #         cos_matix[i,j] = torch.cosine_similarity(prefix[i].unsqueeze(0),caption[j].unsqueeze(0), dim = 1)
    # ax = sns.heatmap(cos_matix)
    # img = ax.get_figure()
    # img.savefig('heatmap/minibatch.jpg')
    # print(cos_matix)

def evaluation(dataloader,model,ImageToPrefixModel,config,args):
    batch_time = AverageMeter('Batch time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(dataloader), [batch_time, losses],
                             prefix='Evaluation: ')
    print_freq = 1000
    
    model.eval()
    device = args.device
    imgid_list,caption_list = [],[]
    prefix_model = ImageToPrefixModel
    model=model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    stop_token_index = tokenizer.encode('.')[0]
    with torch.no_grad():
        for idx,(image,img_id) in enumerate(dataloader):
            end = time.time()
            image = image.to(args.device)
            if args.prefix_type == 'swin':
                    prefix = prefix_model(image)   
            elif args.prefix_type == 'clip':
                prefix = prefix_model.encode_image(image)
            prefix_embed = model.clip_project(prefix).reshape(1,args.prefix_length, -1)
            result_list = generate_prefix(model,prefix_embed=prefix_embed,stop_token_index=stop_token_index)
            imgid_list.append(img_id.item())
            caption_list.append(result_list)
                
            batch_time.update(time.time() - end)
            end = time.time()
      
            if idx % print_freq == 0:
                progress.display(idx)     
    

    result = decode(tokenizer,imgid_list,caption_list)
    # coco_test = coco_caption_eval(config['test_gt_file'], result)
    # print(coco_test.eval,flush = True)
    # path_name = args.checkpoint.split('/')[-2:]
    # with open(f'output/{path_name}_log.txt', "a") as f:
    #     f.write("\n"+json.dumps(coco_test.eval)+"\n")   
    return result   
def eval_only(args,config):
    if args.only_prefix is True:
        args.mapping_type == 'transformer'
        args.prefix_length = args.prefix_length_tr
        args.prefix_length_clip = args.prefix_length_clip_tr
    if args.prefix_type == 'swin':
        prefix_model = SwinModel(args.device)
        args.prefix_length = 50 
        args.prefix_length_clip = 50
    else:
        prefix_model,_ = clip.load('ViT-B/32',device= args.device,jit=False)
        args.prefix_dim = 512
    _,_,test_dataset = create_dataset(args.dataset,config,args)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False,shuffle=True,num_workers=2,pin_memory=True)
    # eval_only(args,config,test_loader,prefix_model)
    mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer,'conv':MappingType.CONV}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                num_layers=args.num_layers, mapping_type=mapping_type)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=args.prefix_dim,
                                num_layers=args.num_layers, mapping_type=mapping_type)
        print("Train both prefix and GPT")
    sys.stdout.flush()
    # flops,params = profile(model,inputs=(1,3,224,224))
    # print('Flops='+str(flops/1000**3)+'G')
    # print('Params='+str(params/1000**3)+'M')
    model = model.to(args.device)
    ckpt = torch.load(args.checkpoint,map_location='cpu')['model']
    model.load_state_dict(ckpt)
    result= evaluation(dataloader=test_loader,
                            model = model,
                            ImageToPrefixModel=prefix_model,
                            args=args,
                            config= config
                            )
    json.dump(result, open(f'output/{args.path}.json', 'w'), indent=4)

def main(args, config):
    eval_only(args, config)
  
    # minibatch_cos(args)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str, default='checkpoints/mlp+gpt2/conv-006.pt')
    parser.add_argument('--prefix_type', type=str, default='swin',help='clip/swin') 
    parser.add_argument('--device',type=str, default='cuda:0')
    parser.add_argument('--dataset', default='coco', help='dataset type:coco/nocaps/cc')
    parser.add_argument('--config',type=str, default='configs/Captioning.yaml')
    parser.add_argument('--path',type=str, help='output result path',default='')
    parser.add_argument('--prefix_dim',type=int,default=1024)
    parser.add_argument('--num_layers',type=int,default=8)
    parser.add_argument('--caption_dir',type=str,default='data/coco/coco_karpathy/coco_karpathy_test_gt1.json')
    # mlp+gpt
    parser.add_argument('--prefix_length', type=int, default=50) # mlp+gpt
    parser.add_argument('--prefix_length_clip', type=int, default=50)# mlp+gpt
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')# mlp+gpt
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    
    # transformer
    parser.add_argument('--prefix_length_tr', type=int, default=50) # transformer
    parser.add_argument('--prefix_length_clip_tr', type=int, default=30) # transformer
    # parser.add_argument('--only_prefix', dest='only_prefix', default=True) # transformer
    # parser.add_argument('--mapping_type', type=str, default='transformer', help='mlp/transformer')
    

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    main(args,config)