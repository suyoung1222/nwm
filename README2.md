export RESULTS_FOLDER=/home/suyoung/Documents/Git/nwm/logs/results
Checkpoints: in ./logs/nwm_cdit_xl/checkpoints
Dataset folders: /home/suyoung/mydata/NWM

테스트데이터 쓸만한거 recon, scand

(gt preparation, one-time)
python isolated_nwm_infer.py     --exp config/nwm_cdit_xl.yaml     --datasets recon,scand     --batch_size 96     --num_workers 12     --eval_type time     --output_dir ${RESULTS_FOLDER}     --gt 1

test
python isolated_nwm_infer.py \
    --exp config/nwm_cdit_xl.yaml \
    --ckp cdit_xl_ego4d_200000 \
    --datasets recon \
    --batch_size 64 \
    --num_workers 12 \
    --eval_type time \
    --output_dir ${RESULTS_FOLDER}

결론:  Recon데이터셋만갖고 one step prediction, trajectory eval, planning eval을 분석하자
cuda 메모리 이슈때문에 더 적은 배치사이즈로 실행시켜야함.
그렇게 하고 트레이닝 데이터가 아닌 주행 데이터 결과가 어떤지 비교해보자 (내가 과거에 쓴 가제보나 아웃도어 이미지 등)