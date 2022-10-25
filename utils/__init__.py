import json
import os
import time
from collections import defaultdict, deque, OrderedDict

import datetime

import numpy as np

import torch
import torch.distributed as dist

from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD

from lib2to3.pgen2 import token
import re
import cv2
import numpy as np
import torch
import torch.nn.functional as nnf
from swin_transformer import SwinTransformer
import os
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
class ScstRewardCriterion(torch.nn.Module):
    CIDER_REWARD_WEIGHT = 1

    def __init__(self, cider_cached_tokens='corpus', baseline_type='greedy'):
        self.CiderD_scorer = CiderD(df=cider_cached_tokens)
        assert baseline_type in ['greedy', 'sample']
        self.baseline_type = baseline_type
        self._cur_score = None
        super().__init__()

    def forward(self, gt_res, greedy_res, sample_res, sample_logprobs):
        batch_size = len(gt_res)
        sample_res_size = len(sample_res)
        seq_per_img = sample_res_size // batch_size

        gen_res = []
        gen_res.extend(sample_res)
        gt_idx = [i // seq_per_img for i in range(sample_res_size)]
        if self.baseline_type == 'greedy':
            assert len(greedy_res) == batch_size
            gen_res.extend(greedy_res)
            gt_idx.extend([i for i in range(batch_size)])

        scores = self._calculate_eval_scores(gen_res, gt_idx, gt_res)

        if self.baseline_type == 'greedy':
            baseline = scores[-batch_size:][:, np.newaxis]
        else:
            sc_ = scores.reshape(batch_size, seq_per_img)
            baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)

        # sample - baseline
        reward = scores[:sample_res_size].reshape(batch_size, seq_per_img)
        self._cur_score = reward.mean()

        reward = reward - baseline
        reward = reward.reshape(sample_res_size)

        reward = torch.as_tensor(reward, device=sample_logprobs.device, dtype=torch.float)
        loss = - sample_logprobs * reward
        loss = loss.mean()
        return loss

    def get_score(self):
        return self._cur_score

    def _calculate_eval_scores(self, gen_res, gt_idx, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [self._wrap_sentence(gen_res[i])]

        gts = OrderedDict()
        gt_res_ = [
            [self._wrap_sentence(gt_res[i][j]) for j in range(len(gt_res[i]))]
                for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[gt_idx[i]]

        res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]
        _, batch_cider_scores = self.CiderD_scorer.compute_score(gts, res_)
        scores = self.CIDER_REWARD_WEIGHT * batch_cider_scores
        return scores

    @classmethod
    def _wrap_sentence(self, s):
        # ensure the sentence ends with <eos> token
        # in order to keep consisitent with cider_cached_tokens
        r = s.strip()
        if r.endswith('.'):
            r = r[:-1]
        r += ' <eos>'
        return r


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, dataset_len=None, epoch_info=None):
        if not header:
            header = ''
        if not dataset_len:
            dataset_len = len(iterable)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(dataset_len))) + 'd'

        _msg = [
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            _msg.append('max mem: {memory:.0f}')
        _msg = self.delimiter.join(_msg)
        MB = 1024.0 * 1024.0
        iterable = iter(iterable)
        train_steps = dataset_len
        if epoch_info:
            start_epoch, end_epoch = epoch_info
            train_steps = (end_epoch - start_epoch) * dataset_len
        for i in range(train_steps):
            obj = next(iterable)
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if epoch_info:
                header = int(i / dataset_len) + start_epoch
                header = 'Train step: [{}]'.format(header)
            log_msg = header + " " + _msg
            if (i % dataset_len) % print_freq == 0 or i == dataset_len - 1:
                eta_seconds = iter_time.global_avg * (dataset_len - i % dataset_len)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i % dataset_len, dataset_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / dataset_len))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def coco_caption_eval(annotation_file, results_file):
    assert os.path.exists(annotation_file)

    # create coco object and coco_result object
    coco = COCO(annotation_file)

    coco_result = coco.loadRes(results_file)
    
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    # for metric, score in coco_eval.eval.items():
    #     print(f'{metric}: {score:.3f}', flush=True)

    return coco_eval

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    caption = caption + '.'

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption

def SwinModel(device):
    model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                embed_dim=128,
                                depths=[ 2, 2, 18, 2 ],
                                num_heads=[ 4, 8, 16, 32 ],
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False)
    model = model.to(device)
    # model_dict = model.state_dict()
    # for key in list(state_dict.keys()):
    #     if key.startswith('vision_encoder.'):
    #         update_key = key.replace('vision_encoder.','')
    #         state_dict[update_key] = state_dict[key]

    # common_dict = {k:v for k, v in state_dict.items() if k in model_dict.keys()}
    # model_dict.update(common_dict)
    ckpt = 'data/coco/swin_pretrain.th'
    state_dict = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(state_dict)
    
    return model

def generate_sin(
    model,
    use_beam,
    stop_token_index,
    args,
    beam_size = 5,
    embed = None,
    entry_length=30,
    temperature=1.0,
    entry_count=1,
    top_p = 0.8
    ):
    
    tokens = None
    scores = None
    device = args.device
    batch_size = args.val_batch_size
    beam_size = args.beam_size
    # if args.use_sparce_mask:
    #     mask = model.get_mask().squeeze()
    # else:
    #     mask = None  
    if use_beam:
        seq_lengths = torch.ones(beam_size,device = device) 
        is_stopped = torch.zeros(beam_size,device= device, dtype = torch.bool)
        if embed is not None:
            generated = embed
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated) 
            logits = outputs.logits # 1 10 50257 
            # logits = logits.view(beam_size,-1)
            logits = logits[:,-1,:]/(temperature if temperature >0 else 1.0) # 1,50257 
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size,-1) # 1,5 1,5  
                generated = generated.expand(beam_size,*generated.shape[1:]) # 5,10,768   
                next_tokens,scores = next_tokens.permute(1,0), scores.squeeze(0) # 5,1 5   
                if tokens is None:
                    tokens = next_tokens # 5,1  b,5,1
                else:
                    tokens = tokens.expand(beam_size,*tokens.shape[1:]) #
                    tokens = torch.cat((tokens,next_tokens),dim=1) #
            else:
                # logits[is_stopped] = -float(np.inf)
                logits[is_stopped] = -float(np.inf) # 5,50257 b,5,v
                logits[is_stopped,0] = 0
                scores_sum = scores[:,None] + logits # 5,50257 = 5,1 + 5,50257 b,5,v = b,5,1 + b,5,v
                # 没有stop token的序列长度加1
                seq_lengths[~is_stopped] += 1 # 5
                scores_sum_average = scores_sum / seq_lengths[:,None] # 5,50257
                scores_sum_average,next_tokens = scores_sum_average.view(-1).topk(beam_size,-1) # 5 5
                next_tokens_source = torch.div(next_tokens, scores_sum.shape[1], rounding_mode='trunc')
                # next_tokens_source = next_tokens // scores_sum.shape[1] # 5
                seq_lengths = seq_lengths[next_tokens_source] # 5
                next_tokens = next_tokens % scores_sum.shape[1] # 5
                next_tokens = next_tokens.unsqueeze(1)  # 5,1
                tokens = tokens[next_tokens_source] 
                tokens = torch.cat((tokens,next_tokens),dim=1) # 5,seq_len
                generated = generated[next_tokens_source] # 5,seq_len,768
                scores = scores_sum_average * seq_lengths # 5
                is_stopped = is_stopped[next_tokens_source] # 5
            next_tokens_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0],1,-1) # 5,1,768
            
            generated = torch.cat((generated,next_tokens_embed),dim=1)# 5,11(token_length),768 
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
        scores = scores / seq_lengths
        output_list = tokens.cpu().numpy()
        order = scores.argsort(descending=True)
        output_list = [output_list[i] for i in order]
        # output_texts=[tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]

    else:
        filter_value = -float('Inf')
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds = generated)
                logits = outputs.logits
                # 取最后一行向量
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
    
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                # if mask is not None:
                #     mask = torch.cat((mask,torch.ones(1,device=mask.device)))
                if stop_token_index == next_token.item():
                    break
            try:
                output_list = list(tokens.squeeze().cpu().numpy())
            except TypeError:
                output_list = [13]
            # output_texts = tokenizer.decode(output_list)
            # # generated_list.append(output_text)
    return output_list

def generate_prefix(
    model,
    stop_token_index,
    prefix_embed = None,
    embed = None,
    entry_length=30,
    temperature=1.0,
    entry_count=1,
    top_p = 0.8
):
    tokens = None
    filter_value = -float('Inf')
    for _ in range(entry_count):
        if prefix_embed is not None:
             # 生成prefix对应的单词
            outputs = model.gpt(inputs_embeds = prefix_embed)
            logits = outputs.logits
            tokens = torch.argmax(logits,-1)
            generated = prefix_embed
        elif embed is not None:
            generated = embed
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds = generated)
            logits = outputs.logits
    
            # 取最后一行向量
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                nnf.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)
            # if tokens is None:
            #     tokens = next_token
            # else:
            tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break
        output_list = list(tokens.squeeze().cpu().numpy())
        # output_texts = tokenizer.decode(output_list)
        # # generated_list.append(output_text)
    return output_list
    
# output_list:list output_texts:list
def decode(tokenizer,imgid_list,token_list):
    output_texts = []
    assert len(imgid_list) == len(token_list)
   
    for i in range(len(imgid_list)):
        caption = tokenizer.decode(token_list[i])
        output_texts.append({'image_id':imgid_list[i],'caption':caption})
        
    return output_texts

def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)