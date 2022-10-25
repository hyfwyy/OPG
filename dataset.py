import os
import json
import random
from collections import Counter
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
from tools import RandomAugment as ra
from tools.RandomAugment import RandomAugment
from utils import pre_caption
from torchvision import transforms
from torchvision.datasets.utils import download_url
from transformers import GPT2Tokenizer
import random
class coco_train(Dataset):
    
    def __init__(self, transform, image_root, ann_rpath, prefix_length,max_words=30):
        self.annotation = []
        for f in ann_rpath:
            self.annotation += json.load(open(f, 'r'))

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('data/gpt2')
        self.prefix_length = prefix_length
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        self.caption = pre_caption(ann['caption'], self.max_words)
        tokens, mask = self.pad_token()
        return image, tokens, mask, self.img_ids[ann['image_id']]
    def pad_token(self):
        # start_token = self.tokenizer.encode('<s>') [27, 82, 29]
        # end_token = self.tokenizer.encode('</s>') [3556, 82, 29]
        tokens = torch.tensor(self.tokenizer.encode(self.caption), dtype=torch.int64)
        padding = self.max_words - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[:self.max_words]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask
class coco_train_scst(Dataset):
    def __init__(self, transform, image_root, ann_rpath, max_words=30, prompt=''):
        self.annotation = []
        self.image_captions_map = {}

        for f in ann_rpath:
            for ann in json.load(open(f, 'r')):
                self.annotation.append(ann)

                if ann['image'] in self.image_captions_map.keys():
                    self.image_captions_map[ann['image']].append(ann['caption'])
                else:
                    self.image_captions_map[ann['image']] = [ann['caption']]

        counter = Counter()
        for _, v in self.image_captions_map.items():
            counter[len(v)] += 1
        print("### image_captions_map, ", counter, flush=True)

        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.prompt = prompt

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # w/o prompt
        captions_gt = [pre_caption(c, self.max_words) for c in self.image_captions_map[ann['image']]]

        return image, random.sample(captions_gt, 5)

    def collate_fn(self, batch_sample):
        batch = []
        for x in zip(*batch_sample):
            batch.append(x)

        image_list, captions_gt_list = batch

        images = torch.stack(image_list)

        return images, captions_gt_list

class coco_caption_eval(Dataset):
    def __init__(self, transform, image_root, ann_rpath):
        self.annotation = json.load(open(ann_rpath, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words=30
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
    
        img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        
        return image, int(img_id)
       
class nocaps_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):   
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/nocaps_test.json'}
        filenames = {'val':'nocaps_val.json','test':'nocaps_test.json'}
        
        download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):  
        
        ann = self.annotation[index] 
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        return image, int(ann['img_id'])    

def create_minibatch(num):
    # ann_dir = 'data/coco/coco_karpathy/coco_karpathy_test_gt1.json'
    ann_dir='data/coco/coco_karpathy/coco_karpathy_test.json'
    image_root = '/raid/datasets/coco2014/'
    minibatch=[]
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    tokenizer = GPT2Tokenizer.from_pretrained('data/gpt2')
    with open(ann_dir,'r') as f:
        ann = json.load(f)

    idx = random.sample(range(0,5000),num)
    mini_cap = [ann[i] for i in idx]
    for item in mini_cap:
        image_path = os.path.join(image_root, str(item['image']))
        image = Image.open(image_path).convert('RGB')   
        image = transform(image)
        tokens = torch.tensor(tokenizer.encode(item['caption'][0]), dtype=torch.int64)
        img_id = item['image'].split('/')[-1].strip('.jpg').split('_')[-1]
        minibatch.append({'image_id':img_id,'image':image,'tokens':tokens})
    return minibatch

def create_dataset(dataset,config,args=None):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0),
                                        interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                    transforms.ToTensor(),
                    normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if dataset == 'coco':
        train_dataset = coco_train(train_transform,config['image_root'],config['train_file'],args.prefix_length,config['max_words'])
        val_dataset = coco_caption_eval(test_transform,config['image_root'],config['val_file'])
        test_dataset = coco_caption_eval(test_transform,config['image_root'],config['test_file'])
        return train_dataset,val_dataset,test_dataset
    elif dataset == 'nocaps':
        val_dataset = nocaps_eval(test_transform,config['image_root'],config['ann_root'],'val')
        # test_dataset = nocaps_eval(test_transform,config['image_root'],config['ann_root'],'test')
        return val_dataset
    elif dataset == 'coco_scst':
        train_dataset = coco_train_scst(train_transform, config['image_root'], config['train_file'],
                                            max_words=config['max_tokens'])
        val_dataset = coco_caption_eval(test_transform, config['image_root'], config['val_file'])
        test_dataset = coco_caption_eval(test_transform, config['image_root'], config['test_file'])

        return train_dataset, val_dataset, test_dataset
        
  
if __name__ == '__main__':
    config = yaml.load(open('configs/nocaps.yaml', 'r'), Loader=yaml.Loader)
    create_dataset('nocaps',config)
    
    