# Run from repo root: cd .../Multi-view_DBRpred && bash scripts/run_evalate_hybrid.sh
# Must match training PSSM location (see train_summary.json / checkpoint args); otherwise
# evaluate_hybrid defaults pssm_root=data_root and PSSM files are not found → zeros → wrong metrics.
python -m mv_dbrpred.evaluate_hybrid \
  --checkpoint /YOUR_PATH/best.pt \
  --data_root /YOUR_PATH/ \
  --pssm_root /YOUR_PATH/ \
  --split test \
  --out_dir /YOUR_PATH/ \
  --num_experts 14 \
  --batch_size 4 