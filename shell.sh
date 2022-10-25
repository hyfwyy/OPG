#!bin/bash
# use contrastive loss
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=3 --output_dir=checkpoints/transformer --lamda=0.1
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=4 --output_dir=checkpoints/transformer4 --lamda=0.1 --device=cuda:0
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=5 --output_dir=checkpoints/transformer5 --lamda=0.1 --device=cuda:0
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=6 --output_dir=checkpoints/transformer6 --lamda=0.1 --device=cuda:0  --checkpoint=checkpoints/transformer6/transformer-005.pt
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=7 --output_dir=checkpoints/transformer7 --lamda=0.1 --device=cuda:0
# python trainv3.py --only_prefix --mapping_type=transformer --use_aux_loss=True --num_layer=8 --output_dir=checkpoints/transformer8 --lamda=0.1 --device=cuda:1

# python trainv3.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gptfozen --only_prefix --mapping_type=mlp
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.1 --mapping_type=mlp --use_aux_loss=True --lamda=0.1
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.2 --mapping_type=mlp --use_aux_loss=True --lamda=0.2
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.3 --mapping_type=mlp --use_aux_loss=True --lamda=0.3
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.4 --mapping_type=mlp --use_aux_loss=True --lamda=0.4
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.5 --mapping_type=mlp --use_aux_loss=True --lamda=0.5

# 重跑一次
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True

# 5508864
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gptfozen-use-aux-loss --mapping_type=mlp --only_prefix --use_aux_loss=True --lamda=0.1

# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-trans6+gpt_train_without_aux_loss --mapping_type=transformer --num_layers=6  --batch_size=30
# 没跑完 200
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-trans6+gpt_train_use_aux_loss --mapping_type=transformer --num_layers=6 --lamda=0.3 --use_aux_loss=True --batch_size=30
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-trans6+gpt_frozen_without_aux_loss --mapping_type=transformer --num_layers=6 --only_prefix

# new  mlp 的 lamda都为0.1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask --mapping_type=mlp --only_prefix --use_sparce_mask=True
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_both_mask_and_loss0.1 --mapping_type=mlp --only_prefix --lamda=0.1 --use_aux_loss=True --use_sparce_mask=True
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gptfozen-use-aux-loss --checkpoint=checkpoints/swinv2-mlp-gptfozen-use-aux-loss/model_latest.pt --mapping_type=mlp --only_prefix --lamda=0.1 --use_aux_loss=True --use_sparce_mask=True --epochs=15
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_use-aux-loss0.1 --mapping_type=mlp --only_prefix --lamda=0.1 --use_aux_loss=True --epochs=15

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.3 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True
 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_train_use-sparce-mask --mapping_type=transformer --num_layers=6 --use_sparce_mask=True --batch_size=30
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_train_both_mask_and_loss0.3 --mapping_type=transformer --num_layers=6 --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True --batch_size=30

# test
# python trainv3.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gptfozen --only_prefix --mapping_type=mlp --eval --checkpoint=checkpoints/swinv2-mlp-gptfozen/mlp-008.pt
# python trainv3.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gptfozen-use-aux-loss --mapping_type=mlp --only_prefix --use_aux_loss=True --lamda=0.1 --eval --checkpoint=checkpoints/swinv2-mlp-gptfozen-use-aux-loss/mlp-012.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask --mapping_type=mlp --only_prefix --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-mlp+gpt_frozen_use-sparce-mask/mlp-008.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp+gpt_frozen_both_mask_and_loss0.1 --mapping_type=mlp --only_prefix --lamda=0.1 --use_aux_loss=True --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-mlp+gpt_frozen_both_mask_and_loss0.1/mlp-009.pt

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask/mlp-008.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-009.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask/transformer-009.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.3 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.3/transformer-009.pt

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gptfozen --only_prefix --mapping_type=mlp --eval --checkpoint=checkpoints/swinv2-mlp-gptfozen/swinv2-mlp-gptfozen-mlp-008.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints/swinv2-mlp-gptfozen-use-aux-loss --mapping_type=mlp --only_prefix --use_aux_loss=True --lamda=0.1 --eval --checkpoint=checkpoints/swinv2-mlp-gptfozen-use-aux-loss/mlp-012-all.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints/swinv2-trans6+gpt_frozen_without_aux_loss --eval --checkpoint=checkpoints/swinv2-trans6+gpt_frozen_without_aux_loss/transformer-009-all.pt --mapping_type=transformer --only_prefix

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True  
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask/mlp-007.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask/mlp-008.pt

# # greedy
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask/mlp-007.pt --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask --mapping_type=mlp --use_sparce_mask=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use-sparce-mask/mlp-008.pt --beam_size=1
# python trainv3.py --device=cuda:0 --eval --checkpoint=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.1/mlp-009.pt --mapping_type=mlp --lamda=0.1 --use_aux_loss=True  --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.1
# python train_scst.py --device=cuda:0 --eval --checkpoint=checkpoints/swinv2-trans6+gpt_frozen_without_aux_loss/transformer-009-all.pt --mapping_type=transformer --only_prefix --num_layers=6 --beam_size=1 --output_dir=checkpoints/swinv2-trans6+gpt_frozen_without_aux_loss
# python train_scst.py --device=cuda:0 --eval --checkpoint=checkpoints/transformer6lamda0.3/swinv2-transformer6lamda0.3-gptfrozen-014.pt --mapping_type=transformer --only_prefix --num_layers=6 --beam_size=1 --output_dir=checkpoints/transformer6lamda0.3
# python train_scst.py --device=cuda:0 --eval --checkpoint=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.1/mlp-009-all.pt --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.1 --mapping_type=mlp --use_aux_loss=True --lamda=0.1 --beam_size=1

# # 200
# # python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True 
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3 --mapping_type=mlp --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True --epochs=15

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-009.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-009.pt --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-010.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-010.pt --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-013.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1/mlp-013.pt --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-009.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-009.pt --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-012.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2 --mapping_type=mlp --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True --epochs=15 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.2/mlp-012.pt --beam_size=1

# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.3 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.2 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.2 --use_aux_loss=True
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_both_mask_and_loss0.1 --mapping_type=transformer --num_layers=6 --only_prefix --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1-up --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1-up/mlp-009.pt
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask/transformer-009.pt --mapping_type=transformer --num_layers=6 --use_sparce_mask=True --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6+gpt_frozen_use-sparce-mask --mapping_type=transformer --only_prefix --num_layers=6 --use_sparce_mask=True
# python train_scst.py --device=cuda:0 --mapping_type=mlp --use_aux_loss=True --lamda=0.1 --eval --checkpoint=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.1/mlp-009-all.pt --save_results=True
# python train_scst.py --device=cuda:0 --mapping_type=transformer --only_prefix --num_layers=6 --use_sparce_mask=True --use_aux_loss=True --lamda=0.3 --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt-frozen_both_mask_and_loss0.3/transformer-009.pt --save_results=True
# python train_scst.py --device=cuda:0 --mapping_type=transformer  --num_layers=6 --use_aux_loss=True --lamda=0.3 --eval --checkpoint=checkpoints_all/swinv2-trans6+gpt_train_use_aux_loss0.3/transformer-009.pt --save_results=True
# python train_scst.py --device=cuda:0 --mapping_type=mlp --use_aux_loss=True --lamda=0.2 --eval --checkpoint=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.2/mlp-007.pt --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.2 --beam_size=1
# python train_scst.py --device=cuda:0 --mapping_type=mlp --use_aux_loss=True --lamda=0.2 --eval --checkpoint=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.3/mlp-008.pt --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.3 --beam_size=1
# python train_scst.py --device=cuda:0 --mapping_type=mlp --use_aux_loss=True --lamda=0.2 --eval --checkpoint=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.4/mlp-008.pt --output_dir=checkpoints/swinv2-mlp-gpt-use_aux_loss-0.4 --beam_size=1

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.1 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.2 --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.2
# python train_scst.py --device=cuda:0 --eval --checkpoint=checkpoints_all/swinv2-trans+gpt-frozen-use-aux-loss0.4/transformer-009.pt --mapping_type=transformer --only_prefix --use_aux_loss=True --lamda=0.4 --beam_size=1
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans+gpt-train-use-aux-loss0.1 --mapping_type=transformer --num_layers=6 --lamda=0.1 --use_aux_loss=True --batch_size_xs=30
# python train_scst.py --device=cuda:0  --output_dir=checkpoints_all/swinv2-trans+gpt-train-use-aux-loss0.2 --mapping_type=transformer --num_layers=6 --lamda=0.2 --use_aux_loss=True --batch_size_xs=30
# python train_scst.py --device=cuda:0 --mapping_type=mlp --use_aux_loss=True --lamda=0.1 --output_dir=checkpoints_all/swinv2-mlp-gpt-use_aux_loss-0.1 --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-use_aux_loss-0.1/mlp-009.pt --beam_size=1

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1 --mapping_type=mlp --use_sparce_mask=True --lamda=0.1 --use_aux_loss=True  
# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3 --mapping_type=mlp --use_sparce_mask=True --lamda=0.3 --use_aux_loss=True  --eval --checkpoint=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.3/mlp-007.pt

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/muge_mlp_gpt_train --dataset=muge --config_xs=configs/Captioning_muge.yaml --mapping_type=mlp --batch_size_xs=20
# 1卡
# python train_scst.py --device=cuda:1 --output_dir=checkpoints_all/muge_trans_gpt_frozen --dataset=muge --config_xs=configs/Captioning_muge.yaml --mapping_type=transformer --only_prefix --batch_size_xs=20

# python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-both_mask_and_loss0.1-v1.0 --mapping_type=mlp --dataset=coco --batch_size_xs=30 --use_aux_loss=True --lamda=0.1  --use_sparce_mask=True --epochs=10
 
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans4-gpt-frozen --mapping_type=transformer --only_prefix --num_layers=4 
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans5-gpt-frozen --mapping_type=transformer --only_prefix --num_layers=5 
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans6-gpt-frozen --mapping_type=transformer --only_prefix --num_layers=6 
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans7-gpt-frozen --mapping_type=transformer --only_prefix --num_layers=7 
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans8-gpt-frozen --mapping_type=transformer --only_prefix --num_layers=8 

python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.1 --mapping_type=mlp --threshold=0.1 --use_sparce_mask=True
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.3 --mapping_type=mlp --threshold=0.3 --use_sparce_mask=True
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.7 --mapping_type=mlp --threshold=0.7 --use_sparce_mask=True
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-mlp-gpt-train-use-mask0.9 --mapping_type=mlp --threshold=0.9 --use_sparce_mask=True

python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-train-use-mask0.7 --mapping_type=transformer  --threshold=0.7 --use_sparce_mask=True
python train_scst.py --device=cuda:0 --output_dir=checkpoints_all/swinv2-trans-gpt-train-use-mask0.9 --mapping_type=transformer  --threshold=0.9 --use_sparce_mask=True

# threshold 改为超参数