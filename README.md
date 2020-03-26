# Cross-GCN
Cross-GCN: Enhancing Graph Convolutional Network with k-Order Feature Interactions

Single Layer

Cross-GCN1_Fix

python run_repeat.py --dataset citeseer --log ./log/single_layer/gcn_mi1_citeseer.log --model gcn_mi1 --epochs 500

python gcn_repeat_analyze.py --dataset citeseer --model gcn_mi1

Cross-GCN1
python run_repeat.py --dataset citeseer --log ./log/single_layer/gcn_mia1_citeseer.log --model gcn_mia1 --epochs 500

python gcn_repeat_analyze.py --dataset citeseer --model gcn_mia1

Two Layers

Cross-GCN_Fix

python run_repeat.py --dataset citeseer --log ./log/two_layers/gcn_mi_citeseer.log --model gcn_mi --hidden1 32 --epochs 500

python gcn_repeat_analyze.py --dataset citeseer --model gcn_mi

Cross-GCN

python run_repeat.py --dataset citeseer --log ./log/two_layers/gcn_mia_citeseer.log --model gcn_mia --hidden1 32 --epochs 500

python gcn_repeat_analyze.py --dataset citeseer --model gcn_mia
