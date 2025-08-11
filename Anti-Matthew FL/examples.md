# adult
nohup python main.py --step_size 0.03 --dataset adult --method EFFL   --eps_g 0.01 --eps_vl 0.03 --eps_vg 0.005  --target_dir_name EFFL_adult -did 2 --max_epoch_stage 750 750 500  --attack_type None > EFFL_adult.out 2>&1 &

# syn
nohup python main.py --step_size 0.03 --dataset synthetic --method EFFL  --eps_g 0.1 --eps_vl 0.01 --eps_vg 0.02  --target_dir_name EFFL_synthetic -did 1 --max_epoch_stage 750 750 500 --attack_type None  > EFFL_synthetic.out 2>&1 &
