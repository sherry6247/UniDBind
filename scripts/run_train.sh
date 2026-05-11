EXPERT_CUDA_VISIBLE_DEVICES=0
cd /YOUR_PATH/

python -m mv_dbrpred.train \
  --data_root /YOUR_PATH/ \
  --esm2_root /YOUR_PATH/ \
  --pssm_root /YOUR_PATH/ \
  --ss_root /YOUR_PATH/ \
  --sasa_root /YOUR_PATH/ \
  --out_dir /YOUR_PATH/ \
  --epochs 50 \
  --batch_size 4 \
  --lr 2e-4 \
  --hidden_dim 256 \
  --topk 0 \
  --patience 15 \
  --seed 42
