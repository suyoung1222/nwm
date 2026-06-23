export RESULTS_FOLDER=/home/suyoung/Documents/Git/nwm/logs/results
Checkpoints: in ./logs/nwm_cdit_xl/checkpoints
Dataset folders: /home/suyoung/mydata/NWM
bag_to_nwm.py --bag ~/dataset/3.bag --dataset_name bunker2024 --traj_name bunker2024 #custom rosbag file

OC-NWM

(gt preparation, one-time)
python isolated_nwm_infer.py     --exp config/nwm_cdit_xl.yaml     --datasets limo1     --batch_size 96     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}     --gt 1

test
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets bunker \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ${RESULTS_FOLDER}

test custom dataset 
/home/suyoung/Documents/limo/agilex_open_class/limo/limo_gazebo_sim/scripts/dataset_mocap/realsense_1/nwm_dataset/limo2_bag
torchrun --nproc_per_node=1 isolated_nwm_infer.py   --exp config/nwm_cdit_xl.yaml   --datasets limo2   --eval_type rollout   --output_dir ./logs/results/limo2
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets limo2 \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ./logs/results/limo2

eval metric result (after gt generation, test)
python isolated_nwm_eval.py \
        --datasets limo2 \
        --gt_dir ${RESULTS_FOLDER}/gt \
        --exp_dir ${RESULTS_FOLDER}/nwm_cdit_xl_cdit_xl_ego4d_200000 \
        --eval_types rollout

결론:  Recon데이터셋만갖고 one step prediction, trajectory eval, planning eval을 분석하자
cuda 메모리 이슈때문에 더 적은 배치사이즈로 실행시켜야함.
그렇게 하고 트레이닝 데이터가 아닌 주행 데이터 결과가 어떤지 비교해보자 (내가 과거에 쓴 가제보나 아웃도어 이미지 등)

rollout limo2 돌린거 gt랑 비교해서 metric결과 확인해보기
똑같은짓 gazebo(limo1,limo5)에서도 돌려보기 -여기까지 완! (사실 도는중이니까 집가서 확인 꼭)
gazebo결과 보고 시뮬레이션 정해서 (gazebo, phoenix, habitat등) 골라서 planning 코드짜기
active mapping베이스라인 코드 정해야할텐ㄷ...
왜 context window size is 4? why not longer?
깃푸씨

Finetuning Result

# gt generation (one-time)
python isolated_nwm_infer.py     --exp config/nwm_cdit_xl.yaml     --datasets bunker2025     --batch_size 96     --num_workers 12     --eval_type rollout     --output_dir ./logs/gt/bunker2025     --gt 1

python isolated_nwm_infer.py     --exp config/nwm_cdit_xl.yaml     --datasets recon     --batch_size 96     --num_workers 12     --eval_type rollout     --output_dir ./logs/gt/recon     --gt 1

# rollout inference comparison
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_finetune_bunker2025_40GB4.yaml \
    --ckp best \
    --datasets bunker2025 \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ./logs/results/ft_bunker2025_infer2

python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets bunker2025 \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ./logs/results/bunker2025


python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl_finetune_bunker2025_40GB4.yaml \
    --ckp best \
    --datasets recon \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ./logs/results/ft_recon 

python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets recon \
    --batch_size 16 \
    --num_workers 12 \
    --eval_type rollout \
    --output_dir ./logs/results/recon

# evaluation metric with predicted rollout
python isolated_nwm_eval.py \
    --datasets bunker2025 \
    --gt_dir ./logs/gt/bunker2025 \
    --exp_dir .logs/nwm_cdit_xl \
    --eval_types rollout

python isolated_nwm_eval.py \
    --datasets bunker2025 \
    --gt_dir ./logs/gt/bunker2025 \
    --exp_dir .logs/nwm_cdit_xl_finetune_bunker2025_40GB4 \
    --eval_types rollout