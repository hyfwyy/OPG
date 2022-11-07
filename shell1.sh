#!bin/bash
# use n pair ms loss
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=3 --output_dir=checkpoints/transformer3lamda0.3 --lamda=0.3 --device=cuda:1
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=4 --output_dir=checkpoints/transformer4lamda0.3 --lamda=0.3 --device=cuda:1
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=5 --output_dir=checkpoints/transformer5lamda0.3 --lamda=0.3 --device=cuda:1
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=6 --output_dir=checkpoints/transformer6lamda0.3 --lamda=0.3 --device=cuda:1  
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=7 --output_dir=checkpoints/transformer7lamda0.3 --lamda=0.3 --device=cuda:1
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=8 --output_dir=checkpoints/transformer8lamda0.3 --lamda=0.3 --device=cuda:1

# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=6 --output_dir=checkpoints/transformer6lamda0.3 --lamda=0.3 --device=cuda:1 --epochs=20
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=6 --output_dir=checkpoints/transformer6lamda0.3 --lamda=0.3 --device=cuda:1 --epochs=15
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=6 --output_dir=checkpoints/transformer6lamda0.3 --lamda=0.3 --device=cuda:1 --epochs=10


# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --only_prefix
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --only_prefix
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --only_prefix --eval --checkpoint=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask/mlp-008.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --only_prefix --eval --checkpoint=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask/mlp-008.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.3 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.2 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.1 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.05 --use_aux_loss=True

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-009.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-009.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-010.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-010.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-013.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-013.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-009.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-009.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-012.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-012.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask/transformer-009.pt --mapping_type=transformer --num_layers=6 --use_sparce_mask=True --beam_size=1

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1-up --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2-up --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3-up --mapping_type=mlp --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1-up --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1-up/mlp-009.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2-up --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2-up/mlp-007.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3-up --mapping_type=mlp --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3-up/mlp-007.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-013.pt --beam_size=1
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-012.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3 --mapping_type=mlp --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3/mlp-012.pt
# python train_scst.py --device=cuda:0 --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask/mlp-007.pt --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask-use-loss0.2 --use_aux_loss=True --lamda=0.2 --use_sparce_mask=True
# python train_scst.py --device=cuda:0 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask-use-loss0.3/mlp-003.pt --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask-use-loss0.3 --use_aux_loss=True --lamda=0.3 --use_sparce_mask=True
# python train_scst.py --device=cuda:0 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask-use-loss0.3/mlp-003.pt --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask-use-loss0.3/model_latest.pt--use_aux_loss=True --lamda=0.3 --use_sparce_mask=True
# python train_scst.py --device=cuda:0 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask-use-loss0.2/mlp-007.pt --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask-use-loss0.2 --use_aux_loss=True --lamda=0.2 --use_sparce_mask=True
# python train_scst.py --device=cuda:1 --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.3/transformer-007.pt  --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.3 --use_sparce_mask=True
# python train_scst.py --device=cuda:1 --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.3/transformer-007.pt --generate_prefix=True --save_results=True --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.3 --use_sparce_mask=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.2 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.2
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.3 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.3
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.4 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.4
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.5 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.5
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.4 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.4
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.5 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.5
# python train_scst.py --device=cuda:1  --output_dir=checkpoints_all/swinv2-trans+gpt-train-use-aux-loss0.4 --mapping_type=transformer --num_layers=6 --lamda=0.4 --use_aux_loss=True --batch_size_xs=30
# python train_scst.py --device=cuda:1  --mapping_type=transformer --num_layers=6 --eval --checkpoint=checkpoints_all/swinv2-trans+gpt-train-use-aux-loss0.1/transformer-006.pt --lamda=0.1 --use_aux_loss=True
# python train_scst.py --device=cuda:1  --mapping_type=transformer --num_layers=6 --eval --checkpoint=checkpoints_all/swinv2-trans+gpt-train-use-aux-loss0.1/transformer-009.pt --lamda=0.1 --use_aux_loss=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp-use_mask-v1.0 --mapping_type=mlp --dataset=coco --batch_size_xs=30  --use_sparce_mask=True --epochs=10

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans4-gpt-train --mapping_type=transformer --num_layers=4 
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans4-gpt-train --mapping_type=transformer --num_layers=4 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans4-gpt-train/transformer-best.pt

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans5-gpt-train --mapping_type=transformer --num_layers=5 
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans5-gpt-train --mapping_type=transformer --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-train/transformer-best.pt

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6-gpt-train --mapping_type=transformer --num_layers=6 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6-gpt-train --mapping_type=transformer --num_layers=6 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans6-gpt-train/transformer-best.pt

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans7-gpt-train --mapping_type=transformer --num_layers=7 
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans7-gpt-train --mapping_type=transformer --num_layers=7 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans7-gpt-train/transformer-best.pt

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans8-gpt-train --mapping_type=transformer --num_layers=8 --checkpoint=checkpoints_all/swinv2-trans8-gpt-train/transformer-best.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans8-gpt-train --mapping_type=transformer --num_layers=8 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans8-gpt-train/transformer-best.pt
# # train for threshold 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.1 --mapping_type=transformer --only_prefix --threshold=0.1 --use_sparce_mask=True --num_layers=5 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.1 --mapping_type=transformer --only_prefix --threshold=0.1 --use_sparce_mask=True --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.1/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.3 --mapping_type=transformer --only_prefix --threshold=0.3 --use_sparce_mask=True --num_layers=5 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.3 --mapping_type=transformer --only_prefix --threshold=0.3 --use_sparce_mask=True --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.3/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.5 --mapping_type=transformer --only_prefix --threshold=0.5 --use_sparce_mask=True --num_layers=5 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.5 --mapping_type=transformer --only_prefix --threshold=0.5 --use_sparce_mask=True --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.5/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.7 --mapping_type=transformer --only_prefix --threshold=0.7 --use_sparce_mask=True --num_layers=5 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.7 --mapping_type=transformer --only_prefix --threshold=0.7 --use_sparce_mask=True --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.7/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.9 --mapping_type=transformer --only_prefix --threshold=0.9 --use_sparce_mask=True --num_layers=5 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.9 --mapping_type=transformer --only_prefix --threshold=0.9 --use_sparce_mask=True --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.9/transformer-best.pt


# eval for threshold 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.1 --mapping_type=transformer --only_prefix --threshold=0.1 --use_sparce_mask=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.1/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.3 --mapping_type=transformer --only_prefix --threshold=0.3 --use_sparce_mask=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.3/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.7 --mapping_type=transformer --only_prefix --threshold=0.7 --use_sparce_mask=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.7/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.9 --mapping_type=transformer --only_prefix --threshold=0.9 --use_sparce_mask=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans-gpt-frozen-use-mask0.9/transformer-best.pt

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-train-use-mask0.1 --mapping_type=transformer  --threshold=0.1 --use_sparce_mask=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans-gpt-train-use-mask0.1/transformer-best.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-train-use-mask0.3 --mapping_type=transformer  --threshold=0.3 --use_sparce_mask=True --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans-gpt-train-use-mask0.3/transformer-best.pt

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.1 --mapping_type=transformer --only_prefix --threshold=0.1 --use_sparce_mask=True --num_layers=5 
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.1 --mapping_type=transformer --only_prefix --threshold=0.1 --use_sparce_mask=True --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.1/transformer-best.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.3 --mapping_type=transformer --only_prefix --threshold=0.3 --use_sparce_mask=True --num_layers=5 
# # python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.3 --mapping_type=transformer --only_prefix --threshold=0.3 --use_sparce_mask=True --num_layers=5 --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans5-gpt-frozen-use-mask0.3/transformer-best.pt
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans2-gpt-frozen --mapping_type=transformer --only_prefix --num_layers=2
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans2-gpt-frozen --mapping_type=transformer --only_prefix --num_layers=2  --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-trans2-gpt-frozen/transformer-best.pt

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt-train --mapping_type=mlp 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp+gpt-train --mapping_type=mlp --eval --beam_size=1 --checkpoint=checkpoints_all/swinv2-mlp+gpt-train/mlp-best.pt

python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-loss0.1 --mapping_type=mlp --use_aux_loss=True --lamda=0.1
python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-loss0.3 --mapping_type=mlp --use_aux_loss=True --lamda=0.3
python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-loss0.5 --mapping_type=mlp --use_aux_loss=True --lamda=0.5
python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-both-mask0.1-loss0.1 --mapping_type=mlp --use_sparce_mask=True --threshold=0.1 --use_aux_loss=True --lamda=0.1
# python train_scst.py --device=cuda:3 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-both-mask0.1-loss0.2 --mapping_type=mlp --use_sparce_mask=True --threshold=0.1 --use_aux_loss=True --lamda=0.2
python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-both-mask0.1-loss0.3 --mapping_type=mlp --use_sparce_mask=True --threshold=0.1 --use_aux_loss=True --lamda=0.3
# python train_scst.py --device=cuda:3 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-both-mask0.1-loss0.4 --mapping_type=mlp --use_sparce_mask=True --threshold=0.1 --use_aux_loss=True --lamda=0.4
python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-mlp+gpt-train-both-mask0.1-loss0.5 --mapping_type=mlp --use_sparce_mask=True --threshold=0.1 --use_aux_loss=True --lamda=0.5

