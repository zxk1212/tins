## ImageNet Four-OOD
  python ./eval_tins_w_init.py \
  --eval-protocol four_ood \
  --in_dataset ImageNet \
  --root-dir ./datasets \
  --gpu 0 \
  --CLIP_ckpt "ViT-B/16" \
  --name "eval_interneg_w_init_four_ood" \
  --inversion-steps 30 \
  --inversion-reg-lambda 0.3 \
  --ood-threshold 0.3 \
  --group-num 5 \
  --ood-number 2000 \
  --extra-text-length 2000



  ## cifar 10 100

    python ./eval_tins_w_init_cifar10_100.py \
  --eval-protocol openood_cifar100 \
  --openood-root /disk1/yangyifeng/icml_2024/OpenOOD \
  --root-dir ./datasets \
  --gpu 7 \
  --CLIP_ckpt "ViT-B/16" \
  --train-shot-per-class 16 \
  --name "interneg_openood_cifar100_16shot" \
  --inversion-steps 30 \
  --inversion-reg-lambda 0.5 \
  --ood-threshold 0.3 \
  --group-num 5 \
  --ood-number 70000 \
  --bank-buffer-size 2000 \
  --extra-text-length 2000


   python ./eval_tins_w_init_cifar10_100.py \
  --eval-protocol openood_cifar10 \
  --openood-root /disk1/yangyifeng/icml_2024/OpenOOD \
  --root-dir ./datasets \
  --gpu 6 \
  --CLIP_ckpt "ViT-B/16" \
  --train-shot-per-class 16 \
  --name "interneg_openood_cifar100_16shot" \
  --inversion-steps 30 \
  --inversion-reg-lambda 0.5 \
  --ood-threshold 0.3 \
  --group-num 5 \
  --ood-number 70000 \
  --bank-buffer-size 2000 \
  --extra-text-length 2000

