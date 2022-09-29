python eval.py --path checkpoints/baseline_res12.pth \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --gpu 0 --eval_dataset MiniImageNet --augment test --data_root ./data
python eval.py --path checkpoints/TSP_res12.pth \
  --eval_all --model_class TSPHead --backbone_class Res12 --num_test_episodes 10000 \
  --gpu 3 --eval_dataset MiniImageNet --t_heads 8 --augment test --data_root ./data
python eval.py --path checkpoints/HMS_res12.pth \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --gpu 0 --eval_dataset MiniImageNet --augment test --data_root ./data
