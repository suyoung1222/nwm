export RESULTS_FOLDER=/home/suyoung/Documents/Git/nwm/logs/results
Checkpoints: in ./logs/nwm_cdit_xl/checkpoints
Dataset folders: /home/suyoung/mydata/NWM

테스트데이터 쓸만한거 recon, scand

(gt preparation, one-time)
python isolated_nwm_infer.py     --exp config/nwm_cdit_xl.yaml     --datasets limo1     --batch_size 96     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}     --gt 1

test
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets limo1 \
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
똑같은짓 gazebo(limo1,limo5)에서도 돌려보기
gazebo결과 보고 시뮬레이션 정해서 (gazebo, phoenix, habitat등) 골라서 planning 코드짜기
깃푸씨