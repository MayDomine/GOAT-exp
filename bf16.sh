python  arxiv_ERM_ns.py \
    --dataset ogbn-arxiv \
    --lr 1e-5 \
    --batch_size 1024 \
    --test_batch_size 256 \
    --hidden_dim 128 \
    --global_dim 128 \
    --test_freq 1 \
    --num_workers 4 \
    --conv_type full \
    --num_heads 4 \
    --num_centroids 4096 \
    --data_root ./data/ogb \
	--dtype bf16 \
    --linkx_data_root ./data/linkx
