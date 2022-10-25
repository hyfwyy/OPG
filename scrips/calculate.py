from utils import coco_caption_eval
import yaml
config_path = 'Captioning.yaml'
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
results = 'output/mlp-005_test_results.json'

coco_test = coco_caption_eval(config['test_gt_file'], results)
log_stats = {**{f'test_{k}': v for k, v in coco_test.eval.items()}}
print(log_stats, flush=True)