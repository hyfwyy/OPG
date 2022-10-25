import json
rpath = 'data/coco/coco_karpathy/coco_karpathy_val_gt.json'
wpath = 'data/coco/coco_karpathy/coco_karpathy_val_gt1.json'

with open(rpath,'r') as f:
    file = json.load(f)
    
ls = []
num=0
for item in file['annotations']:
    if item['caption'].split()[-1][-1] != '.':
        item['caption'] = item['caption']+'.'
        num+=1
    ls.append(item)
assert len(file['annotations']) == len(ls)
file['annotations'] = ls
with open(wpath,'w') as f1:
    json.dump(file,f1)
print(f'{num} captions are modefied.')