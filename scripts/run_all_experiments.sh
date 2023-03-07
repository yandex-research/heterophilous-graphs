python train.py --name ResNet_l1 --dataset roman-empire --model ResNet --num_layers 1 --device cuda:0
python train.py --name ResNet_l2 --dataset roman-empire --model ResNet --num_layers 2 --device cuda:0
python train.py --name ResNet_l3 --dataset roman-empire --model ResNet --num_layers 3 --device cuda:0
python train.py --name ResNet_l4 --dataset roman-empire --model ResNet --num_layers 4 --device cuda:0
python train.py --name ResNet_l5 --dataset roman-empire --model ResNet --num_layers 5 --device cuda:0

python train.py --name ResNet_SGC_l1 --dataset roman-empire --model ResNet --num_layers 1 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l2 --dataset roman-empire --model ResNet --num_layers 2 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l3 --dataset roman-empire --model ResNet --num_layers 3 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l4 --dataset roman-empire --model ResNet --num_layers 4 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l5 --dataset roman-empire --model ResNet --num_layers 5 --use_sgc_features --device cuda:0

python train.py --name ResNet_adj_l1 --dataset roman-empire --model ResNet --num_layers 1 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l2 --dataset roman-empire --model ResNet --num_layers 2 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l3 --dataset roman-empire --model ResNet --num_layers 3 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l4 --dataset roman-empire --model ResNet --num_layers 4 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l5 --dataset roman-empire --model ResNet --num_layers 5 --use_adjacency_features --device cuda:0

python train.py --name GCN_l1 --dataset roman-empire --model GCN --num_layers 1 --device cuda:0
python train.py --name GCN_l2 --dataset roman-empire --model GCN --num_layers 2 --device cuda:0
python train.py --name GCN_l3 --dataset roman-empire --model GCN --num_layers 3 --device cuda:0
python train.py --name GCN_l4 --dataset roman-empire --model GCN --num_layers 4 --device cuda:0
python train.py --name GCN_l5 --dataset roman-empire --model GCN --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset roman-empire --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset roman-empire --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset roman-empire --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset roman-empire --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset roman-empire --model SAGE --num_layers 5 --device cuda:0

python train.py --name GAT_l1 --dataset roman-empire --model GAT --num_layers 1 --device cuda:0
python train.py --name GAT_l2 --dataset roman-empire --model GAT --num_layers 2 --device cuda:0
python train.py --name GAT_l3 --dataset roman-empire --model GAT --num_layers 3 --device cuda:0
python train.py --name GAT_l4 --dataset roman-empire --model GAT --num_layers 4 --device cuda:0
python train.py --name GAT_l5 --dataset roman-empire --model GAT --num_layers 5 --device cuda:0

python train.py --name GAT_sep_l1 --dataset roman-empire --model GAT-sep --num_layers 1 --device cuda:0
python train.py --name GAT_sep_l2 --dataset roman-empire --model GAT-sep --num_layers 2 --device cuda:0
python train.py --name GAT_sep_l3 --dataset roman-empire --model GAT-sep --num_layers 3 --device cuda:0
python train.py --name GAT_sep_l4 --dataset roman-empire --model GAT-sep --num_layers 4 --device cuda:0
python train.py --name GAT_sep_l5 --dataset roman-empire --model GAT-sep --num_layers 5 --device cuda:0

python train.py --name GT_l1 --dataset roman-empire --model GT --num_layers 1 --device cuda:0
python train.py --name GT_l2 --dataset roman-empire --model GT --num_layers 2 --device cuda:0
python train.py --name GT_l3 --dataset roman-empire --model GT --num_layers 3 --device cuda:0
python train.py --name GT_l4 --dataset roman-empire --model GT --num_layers 4 --device cuda:0
python train.py --name GT_l5 --dataset roman-empire --model GT --num_layers 5 --device cuda:0

python train.py --name GT_sep_l1 --dataset roman-empire --model GT-sep --num_layers 1 --device cuda:0
python train.py --name GT_sep_l2 --dataset roman-empire --model GT-sep --num_layers 2 --device cuda:0
python train.py --name GT_sep_l3 --dataset roman-empire --model GT-sep --num_layers 3 --device cuda:0
python train.py --name GT_sep_l4 --dataset roman-empire --model GT-sep --num_layers 4 --device cuda:0
python train.py --name GT_sep_l5 --dataset roman-empire --model GT-sep --num_layers 5 --device cuda:0





python train.py --name ResNet_l1 --dataset amazon-ratings --model ResNet --num_layers 1 --device cuda:0
python train.py --name ResNet_l2 --dataset amazon-ratings --model ResNet --num_layers 2 --device cuda:0
python train.py --name ResNet_l3 --dataset amazon-ratings --model ResNet --num_layers 3 --device cuda:0
python train.py --name ResNet_l4 --dataset amazon-ratings --model ResNet --num_layers 4 --device cuda:0
python train.py --name ResNet_l5 --dataset amazon-ratings --model ResNet --num_layers 5 --device cuda:0

python train.py --name ResNet_SGC_l1 --dataset amazon-ratings --model ResNet --num_layers 1 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l2 --dataset amazon-ratings --model ResNet --num_layers 2 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l3 --dataset amazon-ratings --model ResNet --num_layers 3 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l4 --dataset amazon-ratings --model ResNet --num_layers 4 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l5 --dataset amazon-ratings --model ResNet --num_layers 5 --use_sgc_features --device cuda:0

python train.py --name ResNet_adj_l1 --dataset amazon-ratings --model ResNet --num_layers 1 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l2 --dataset amazon-ratings --model ResNet --num_layers 2 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l3 --dataset amazon-ratings --model ResNet --num_layers 3 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l4 --dataset amazon-ratings --model ResNet --num_layers 4 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l5 --dataset amazon-ratings --model ResNet --num_layers 5 --use_adjacency_features --device cuda:0

python train.py --name GCN_l1 --dataset amazon-ratings --model GCN --num_layers 1 --device cuda:0
python train.py --name GCN_l2 --dataset amazon-ratings --model GCN --num_layers 2 --device cuda:0
python train.py --name GCN_l3 --dataset amazon-ratings --model GCN --num_layers 3 --device cuda:0
python train.py --name GCN_l4 --dataset amazon-ratings --model GCN --num_layers 4 --device cuda:0
python train.py --name GCN_l5 --dataset amazon-ratings --model GCN --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset amazon-ratings --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset amazon-ratings --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset amazon-ratings --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset amazon-ratings --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset amazon-ratings --model SAGE --num_layers 5 --device cuda:0

python train.py --name GAT_l1 --dataset amazon-ratings --model GAT --num_layers 1 --device cuda:0
python train.py --name GAT_l2 --dataset amazon-ratings --model GAT --num_layers 2 --device cuda:0
python train.py --name GAT_l3 --dataset amazon-ratings --model GAT --num_layers 3 --device cuda:0
python train.py --name GAT_l4 --dataset amazon-ratings --model GAT --num_layers 4 --device cuda:0
python train.py --name GAT_l5 --dataset amazon-ratings --model GAT --num_layers 5 --device cuda:0

python train.py --name GAT_sep_l1 --dataset amazon-ratings --model GAT-sep --num_layers 1 --device cuda:0
python train.py --name GAT_sep_l2 --dataset amazon-ratings --model GAT-sep --num_layers 2 --device cuda:0
python train.py --name GAT_sep_l3 --dataset amazon-ratings --model GAT-sep --num_layers 3 --device cuda:0
python train.py --name GAT_sep_l4 --dataset amazon-ratings --model GAT-sep --num_layers 4 --device cuda:0
python train.py --name GAT_sep_l5 --dataset amazon-ratings --model GAT-sep --num_layers 5 --device cuda:0

python train.py --name GT_l1 --dataset amazon-ratings --model GT --num_layers 1 --device cuda:0
python train.py --name GT_l2 --dataset amazon-ratings --model GT --num_layers 2 --device cuda:0
python train.py --name GT_l3 --dataset amazon-ratings --model GT --num_layers 3 --device cuda:0
python train.py --name GT_l4 --dataset amazon-ratings --model GT --num_layers 4 --device cuda:0
python train.py --name GT_l5 --dataset amazon-ratings --model GT --num_layers 5 --device cuda:0

python train.py --name GT_sep_l1 --dataset amazon-ratings --model GT-sep --num_layers 1 --device cuda:0
python train.py --name GT_sep_l2 --dataset amazon-ratings --model GT-sep --num_layers 2 --device cuda:0
python train.py --name GT_sep_l3 --dataset amazon-ratings --model GT-sep --num_layers 3 --device cuda:0
python train.py --name GT_sep_l4 --dataset amazon-ratings --model GT-sep --num_layers 4 --device cuda:0
python train.py --name GT_sep_l5 --dataset amazon-ratings --model GT-sep --num_layers 5 --device cuda:0





python train.py --name ResNet_l1 --dataset minesweeper --model ResNet --num_layers 1 --device cuda:0
python train.py --name ResNet_l2 --dataset minesweeper --model ResNet --num_layers 2 --device cuda:0
python train.py --name ResNet_l3 --dataset minesweeper --model ResNet --num_layers 3 --device cuda:0
python train.py --name ResNet_l4 --dataset minesweeper --model ResNet --num_layers 4 --device cuda:0
python train.py --name ResNet_l5 --dataset minesweeper --model ResNet --num_layers 5 --device cuda:0

python train.py --name ResNet_SGC_l1 --dataset minesweeper --model ResNet --num_layers 1 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l2 --dataset minesweeper --model ResNet --num_layers 2 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l3 --dataset minesweeper --model ResNet --num_layers 3 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l4 --dataset minesweeper --model ResNet --num_layers 4 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l5 --dataset minesweeper --model ResNet --num_layers 5 --use_sgc_features --device cuda:0

python train.py --name ResNet_adj_l1 --dataset minesweeper --model ResNet --num_layers 1 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l2 --dataset minesweeper --model ResNet --num_layers 2 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l3 --dataset minesweeper --model ResNet --num_layers 3 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l4 --dataset minesweeper --model ResNet --num_layers 4 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l5 --dataset minesweeper --model ResNet --num_layers 5 --use_adjacency_features --device cuda:0

python train.py --name GCN_l1 --dataset minesweeper --model GCN --num_layers 1 --device cuda:0
python train.py --name GCN_l2 --dataset minesweeper --model GCN --num_layers 2 --device cuda:0
python train.py --name GCN_l3 --dataset minesweeper --model GCN --num_layers 3 --device cuda:0
python train.py --name GCN_l4 --dataset minesweeper --model GCN --num_layers 4 --device cuda:0
python train.py --name GCN_l5 --dataset minesweeper --model GCN --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset minesweeper --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset minesweeper --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset minesweeper --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset minesweeper --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset minesweeper --model SAGE --num_layers 5 --device cuda:0

python train.py --name GAT_l1 --dataset minesweeper --model GAT --num_layers 1 --device cuda:0
python train.py --name GAT_l2 --dataset minesweeper --model GAT --num_layers 2 --device cuda:0
python train.py --name GAT_l3 --dataset minesweeper --model GAT --num_layers 3 --device cuda:0
python train.py --name GAT_l4 --dataset minesweeper --model GAT --num_layers 4 --device cuda:0
python train.py --name GAT_l5 --dataset minesweeper --model GAT --num_layers 5 --device cuda:0

python train.py --name GAT_sep_l1 --dataset minesweeper --model GAT-sep --num_layers 1 --device cuda:0
python train.py --name GAT_sep_l2 --dataset minesweeper --model GAT-sep --num_layers 2 --device cuda:0
python train.py --name GAT_sep_l3 --dataset minesweeper --model GAT-sep --num_layers 3 --device cuda:0
python train.py --name GAT_sep_l4 --dataset minesweeper --model GAT-sep --num_layers 4 --device cuda:0
python train.py --name GAT_sep_l5 --dataset minesweeper --model GAT-sep --num_layers 5 --device cuda:0

python train.py --name GT_l1 --dataset minesweeper --model GT --num_layers 1 --device cuda:0
python train.py --name GT_l2 --dataset minesweeper --model GT --num_layers 2 --device cuda:0
python train.py --name GT_l3 --dataset minesweeper --model GT --num_layers 3 --device cuda:0
python train.py --name GT_l4 --dataset minesweeper --model GT --num_layers 4 --device cuda:0
python train.py --name GT_l5 --dataset minesweeper --model GT --num_layers 5 --device cuda:0

python train.py --name GT_sep_l1 --dataset minesweeper --model GT-sep --num_layers 1 --device cuda:0
python train.py --name GT_sep_l2 --dataset minesweeper --model GT-sep --num_layers 2 --device cuda:0
python train.py --name GT_sep_l3 --dataset minesweeper --model GT-sep --num_layers 3 --device cuda:0
python train.py --name GT_sep_l4 --dataset minesweeper --model GT-sep --num_layers 4 --device cuda:0
python train.py --name GT_sep_l5 --dataset minesweeper --model GT-sep --num_layers 5 --device cuda:0





python train.py --name ResNet_l1 --dataset tolokers --model ResNet --num_layers 1 --device cuda:0
python train.py --name ResNet_l2 --dataset tolokers --model ResNet --num_layers 2 --device cuda:0
python train.py --name ResNet_l3 --dataset tolokers --model ResNet --num_layers 3 --device cuda:0
python train.py --name ResNet_l4 --dataset tolokers --model ResNet --num_layers 4 --device cuda:0
python train.py --name ResNet_l5 --dataset tolokers --model ResNet --num_layers 5 --device cuda:0

python train.py --name ResNet_SGC_l1 --dataset tolokers --model ResNet --num_layers 1 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l2 --dataset tolokers --model ResNet --num_layers 2 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l3 --dataset tolokers --model ResNet --num_layers 3 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l4 --dataset tolokers --model ResNet --num_layers 4 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l5 --dataset tolokers --model ResNet --num_layers 5 --use_sgc_features --device cuda:0

python train.py --name ResNet_adj_l1 --dataset tolokers --model ResNet --num_layers 1 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l2 --dataset tolokers --model ResNet --num_layers 2 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l3 --dataset tolokers --model ResNet --num_layers 3 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l4 --dataset tolokers --model ResNet --num_layers 4 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l5 --dataset tolokers --model ResNet --num_layers 5 --use_adjacency_features --device cuda:0

python train.py --name GCN_l1 --dataset tolokers --model GCN --num_layers 1 --device cuda:0
python train.py --name GCN_l2 --dataset tolokers --model GCN --num_layers 2 --device cuda:0
python train.py --name GCN_l3 --dataset tolokers --model GCN --num_layers 3 --device cuda:0
python train.py --name GCN_l4 --dataset tolokers --model GCN --num_layers 4 --device cuda:0
python train.py --name GCN_l5 --dataset tolokers --model GCN --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset tolokers --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset tolokers --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset tolokers --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset tolokers --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset tolokers --model SAGE --num_layers 5 --device cuda:0

python train.py --name GAT_l1 --dataset tolokers --model GAT --num_layers 1 --device cuda:0
python train.py --name GAT_l2 --dataset tolokers --model GAT --num_layers 2 --device cuda:0
python train.py --name GAT_l3 --dataset tolokers --model GAT --num_layers 3 --device cuda:0
python train.py --name GAT_l4 --dataset tolokers --model GAT --num_layers 4 --device cuda:0
python train.py --name GAT_l5 --dataset tolokers --model GAT --num_layers 5 --device cuda:0

python train.py --name GAT_sep_l1 --dataset tolokers --model GAT-sep --num_layers 1 --device cuda:0
python train.py --name GAT_sep_l2 --dataset tolokers --model GAT-sep --num_layers 2 --device cuda:0
python train.py --name GAT_sep_l3 --dataset tolokers --model GAT-sep --num_layers 3 --device cuda:0
python train.py --name GAT_sep_l4 --dataset tolokers --model GAT-sep --num_layers 4 --device cuda:0
python train.py --name GAT_sep_l5 --dataset tolokers --model GAT-sep --num_layers 5 --device cuda:0

python train.py --name GT_l1 --dataset tolokers --model GT --num_layers 1 --device cuda:0
python train.py --name GT_l2 --dataset tolokers --model GT --num_layers 2 --device cuda:0
python train.py --name GT_l3 --dataset tolokers --model GT --num_layers 3 --device cuda:0
python train.py --name GT_l4 --dataset tolokers --model GT --num_layers 4 --device cuda:0
python train.py --name GT_l5 --dataset tolokers --model GT --num_layers 5 --device cuda:0

python train.py --name GT_sep_l1 --dataset tolokers --model GT-sep --num_layers 1 --device cuda:0
python train.py --name GT_sep_l2 --dataset tolokers --model GT-sep --num_layers 2 --device cuda:0
python train.py --name GT_sep_l3 --dataset tolokers --model GT-sep --num_layers 3 --device cuda:0
python train.py --name GT_sep_l4 --dataset tolokers --model GT-sep --num_layers 4 --device cuda:0
python train.py --name GT_sep_l5 --dataset tolokers --model GT-sep --num_layers 5 --device cuda:0





python train.py --name ResNet_l1 --dataset questions --model ResNet --num_layers 1 --device cuda:0
python train.py --name ResNet_l2 --dataset questions --model ResNet --num_layers 2 --device cuda:0
python train.py --name ResNet_l3 --dataset questions --model ResNet --num_layers 3 --device cuda:0
python train.py --name ResNet_l4 --dataset questions --model ResNet --num_layers 4 --device cuda:0
python train.py --name ResNet_l5 --dataset questions --model ResNet --num_layers 5 --device cuda:0

python train.py --name ResNet_SGC_l1 --dataset questions --model ResNet --num_layers 1 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l2 --dataset questions --model ResNet --num_layers 2 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l3 --dataset questions --model ResNet --num_layers 3 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l4 --dataset questions --model ResNet --num_layers 4 --use_sgc_features --device cuda:0
python train.py --name ResNet_SGC_l5 --dataset questions --model ResNet --num_layers 5 --use_sgc_features --device cuda:0

python train.py --name ResNet_adj_l1 --dataset questions --model ResNet --num_layers 1 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l2 --dataset questions --model ResNet --num_layers 2 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l3 --dataset questions --model ResNet --num_layers 3 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l4 --dataset questions --model ResNet --num_layers 4 --use_adjacency_features --device cuda:0
python train.py --name ResNet_adj_l5 --dataset questions --model ResNet --num_layers 5 --use_adjacency_features --device cuda:0

python train.py --name GCN_l1 --dataset questions --model GCN --num_layers 1 --device cuda:0
python train.py --name GCN_l2 --dataset questions --model GCN --num_layers 2 --device cuda:0
python train.py --name GCN_l3 --dataset questions --model GCN --num_layers 3 --device cuda:0
python train.py --name GCN_l4 --dataset questions --model GCN --num_layers 4 --device cuda:0
python train.py --name GCN_l5 --dataset questions --model GCN --num_layers 5 --device cuda:0

python train.py --name SAGE_l1 --dataset questions --model SAGE --num_layers 1 --device cuda:0
python train.py --name SAGE_l2 --dataset questions --model SAGE --num_layers 2 --device cuda:0
python train.py --name SAGE_l3 --dataset questions --model SAGE --num_layers 3 --device cuda:0
python train.py --name SAGE_l4 --dataset questions --model SAGE --num_layers 4 --device cuda:0
python train.py --name SAGE_l5 --dataset questions --model SAGE --num_layers 5 --device cuda:0

python train.py --name GAT_l1 --dataset questions --model GAT --num_layers 1 --device cuda:0
python train.py --name GAT_l2 --dataset questions --model GAT --num_layers 2 --device cuda:0
python train.py --name GAT_l3 --dataset questions --model GAT --num_layers 3 --device cuda:0
python train.py --name GAT_l4 --dataset questions --model GAT --num_layers 4 --device cuda:0
python train.py --name GAT_l5 --dataset questions --model GAT --num_layers 5 --device cuda:0

python train.py --name GAT_sep_l1 --dataset questions --model GAT-sep --num_layers 1 --device cuda:0
python train.py --name GAT_sep_l2 --dataset questions --model GAT-sep --num_layers 2 --device cuda:0
python train.py --name GAT_sep_l3 --dataset questions --model GAT-sep --num_layers 3 --device cuda:0
python train.py --name GAT_sep_l4 --dataset questions --model GAT-sep --num_layers 4 --device cuda:0
python train.py --name GAT_sep_l5 --dataset questions --model GAT-sep --num_layers 5 --device cuda:0

python train.py --name GT_l1 --dataset questions --model GT --num_layers 1 --device cuda:0
python train.py --name GT_l2 --dataset questions --model GT --num_layers 2 --device cuda:0
python train.py --name GT_l3 --dataset questions --model GT --num_layers 3 --device cuda:0
python train.py --name GT_l4 --dataset questions --model GT --num_layers 4 --device cuda:0
python train.py --name GT_l5 --dataset questions --model GT --num_layers 5 --device cuda:0

python train.py --name GT_sep_l1 --dataset questions --model GT-sep --num_layers 1 --device cuda:0
python train.py --name GT_sep_l2 --dataset questions --model GT-sep --num_layers 2 --device cuda:0
python train.py --name GT_sep_l3 --dataset questions --model GT-sep --num_layers 3 --device cuda:0
python train.py --name GT_sep_l4 --dataset questions --model GT-sep --num_layers 4 --device cuda:0
python train.py --name GT_sep_l5 --dataset questions --model GT-sep --num_layers 5 --device cuda:0
