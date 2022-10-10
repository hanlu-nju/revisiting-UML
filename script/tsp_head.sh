python train.py --eval_all --unsupervised --batch_size 32 --augment 'AMDIM' \
  --num_tasks 256 --max_epoch 200 --model_class TSPHead --backbone_class Res12 \
  --dataset MiniImageNet --way 5 --shot 1 --query 5 --eval_query 15 \
  --temperature 1 --temperature2 1 --lr 0.03 --lr_mul 10 --lr_scheduler cosine \
  --gpu 0 --eval_interval 2 --similarity sns --t_heads 8


python train.py --eval_all --unsupervised --batch_size 64 --augment 'AMDIM' \
  --num_tasks 256 --max_epoch 200 --model_class TSPHead --backbone_class Res12 \
  --dataset MiniImageNet --way 5 --shot 1 --query 5 --eval_query 15 \
  --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 10 --lr_scheduler cosine \
  --gpu 0 --eval_interval 2 --similarity sns --t_heads 8
