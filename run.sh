# Ours

python3 src/run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 20 --save_step 1000 --save_best --save_last --reweight_groups --resample --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --train_epoch_size 2500 --small_val_epoch_size 500  --robust_step_size 0.2 --log_dir multiNLI_CL_wd-0_lr-2e-05_adj-0

python3 src/run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 3 --save_step 1000 --save_best --save_last --reweight_groups --resample --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --train_epoch_size 2500 --small_val_epoch_size 500  --robust_step_size 1 --log_dir multiNLI_3_CL_wd-0_lr-2e-05_adj-0

python3 src/run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.0001 --lr 0.0001 --batch_size 128 --n_epochs 50 --save_step 1000 --save_best --save_last --reweight_groups --resample --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --train_epoch_size 5000 --small_val_epoch_size 1000  --robust_step_size 1 --log_dir celebA_CL_wd-0.0001_lr-0.0001_adj-0

python3 src/run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 --batch_size 128 --n_epochs 50 --save_step 1000 --save_best --save_last --reweight_groups --resample --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --train_epoch_size 5000 --small_val_epoch_size 1000  --robust_step_size 1  --log_dir celebA_CL_wd-0.1_lr-1e-05_adj-0

python3 src/run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.0001 --lr 0.001 --batch_size 128 --n_epochs 300 --save_step 1000 --save_best --save_last --reweight_groups --resample --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --train_epoch_size 5000 --small_val_epoch_size 1000  --robust_step_size 1  --log_dir waterbirds_CL_wd-0.0001_lr-0.001_adj-0

python3 src/run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 1 --lr 1e-05 --batch_size 128 --n_epochs 100 --save_step 1000 --save_best --save_last --reweight_groups --resample --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --train_epoch_size 2500 --small_val_epoch_size 500  --robust_step_size 0.2 --log_dir waterbirds_CL_wd-1_lr-1e-05_adj-0

# GDRO

python3 src/run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.0001 --lr 0.0001 --batch_size 128 --n_epochs 50 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --log_dir 	celebA_DRO_wd-0.0001_lr-0.0001_adj-0

python3 src/run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 --batch_size 128 --n_epochs 50 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --log_dir 	celebA_DRO_wd-0.1_lr-1e-05_adj-0

python3 src/run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.0001 --lr 0.001 --batch_size 128 --n_epochs 300 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --log_dir waterbirds_DRO_wd-0.0001_lr-0.001_adj-0

python3 src/run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 1 --lr 1e-05 --batch_size 128 --n_epochs 300 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --log_dir waterbirds_DRO_wd-1_lr-1e-05_adj-0

python3 src/run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 20 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --log_dir multiNLI_DRO_wd-0_lr-2e-05_adj-0

python3 src/run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 3 --save_step 1000 --save_best --save_last --reweight_groups --robust --alpha 0.01 --gamma 0.1 --generalization_adjustment 0 --log_dir 	multiNLI_3_DRO_wd-0_lr-2e-05_adj-0

# ERM

python3 src/run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.0001 --lr 0.0001 --batch_size 128 --n_epochs 50 --save_step 1000 --save_best --save_last --log_dir 	celebA_ERM_wd-0.0001_lr-0.0001

python3 src/run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --model resnet50 --weight_decay 0.1 --lr 1e-05 --batch_size 128 --n_epochs 50 --save_step 1000 --save_best --save_last --log_dir 	celebA_ERM_wd-0.1_lr-1e-05

python3 src/run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 0.0001 --lr 0.001 --batch_size 128 --n_epochs 300 --save_step 1000 --save_best --save_last --log_dir waterbirds_ERM_wd-0.0001_lr-0.001 

python3 src/run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --model resnet50 --weight_decay 1 --lr 1e-05 --batch_size 128 --n_epochs 300 --save_step 1000 --save_best --save_last --log_dir waterbirds_ERM_wd-1_lr-1e-05

python3 src/run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 20 --save_step 1000 --save_best --save_last --log_dir multiNLI_ERM_wd-0_lr-2e-05

python3 src/run_expt.py -s confounder -d MultiNLI -t gold_label_random -c sentence2_has_negation --model bert --weight_decay 0 --lr 2e-05 --batch_size 32 --n_epochs 3 --save_step 1000 --save_best --save_last --log_dir multiNLI_3_ERM_wd-0_lr-2e-05
