# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --use_aux_loss=True --lamda=0.1 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-005.pt --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --use_aux_loss=True --lamda=0.1 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-008.pt --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --use_aux_loss=True --lamda=0.1 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-009.pt --beam_size=1

# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-trans4-gpt-frozen/transformer-best.pt --mapping_type=transformer --only_prefix --num_layers=4 --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-trans4-gpt-frozen
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen/transformer-best.pt --mapping_type=transformer --only_prefix --num_layers=5 --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-trans6-gpt-frozen/transformer-best.pt --mapping_type=transformer --only_prefix --num_layers=6 --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-trans6-gpt-frozen
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-trans7-gpt-frozen/transformer-best.pt --mapping_type=transformer --only_prefix --num_layers=7 --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-trans7-gpt-frozen
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-trans8-gpt-frozen/transformer-best.pt --mapping_type=transformer --only_prefix --num_layers=8 --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-trans8-gpt-frozen

# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.1/mlp-best.pt --mapping_type=mlp --threshold=0.1 --use_sparce_mask=True --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.1
# 还没跑完
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.3/mlp-best.pt --mapping_type=mlp --threshold=0.3 --use_sparce_mask=True --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.3
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.7/mlp-best.pt --mapping_type=mlp --threshold=0.7 --use_sparce_mask=True --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.7
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.9/mlp-best.pt --mapping_type=mlp --threshold=0.9 --use_sparce_mask=True --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.9

# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-trans-gpt-train-use-mask0.7/transformer-best.pt --mapping_type=transformer  --threshold=0.7 --use_sparce_mask=True --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-trans-gpt-train-use-mask0.7
# python train_scst.py --device=cuda:1 --checkpoint=checkpoints_all/swinv2-trans-gpt-train-use-mask0.9/transformer-best.pt --mapping_type=transformer  --threshold=0.9 --use_sparce_mask=True --eval --beam_size=1 --output_dir=checkpoints_all/swinv2-trans-gpt-train-use-mask0.9

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans5-gpt-train --mapping_type=transformer --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-train/transformer-best.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6-gpt-train --mapping_type=transformer --num_layers=6 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans6-gpt-train/transformer-best.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans7-gpt-train --mapping_type=transformer --num_layers=7 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans7-gpt-train/transformer-best.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans8-gpt-train --mapping_type=transformer --num_layers=8 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans8-gpt-train/transformer-best.pt
