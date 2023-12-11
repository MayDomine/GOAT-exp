import torch
class RelativePositionEmbedding(torch.nn.Module):
    """ Relative Position Embedding <https://arxiv.org/abs/1803.02155>

    Args:
        num_heads (int): number of heads used in attention module.
        num_buckets (int, optional): Defaults to 32.
        max_distance (int, optional): Defaults to 128.
        bidirectional (bool, optional): Defaults to False.
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): Defaults to 0.0.
        init_std (float, optional): Defaults to 1.
    """

    def __init__(self, num_heads : int, 
                       num_buckets : int = 32, 
                       max_distance : int = 128, 
                       bidirectional : bool = False, 
                       dtype = torch.half,
                       init_mean : float = 0.0,
                       init_std : float = 1,
                    ):

        super().__init__()

        self.relative_attention_bias = bmt.DistributedParameter(
            torch.empty(num_buckets, num_heads, dtype = dtype), 
            init_method = bmt.ParameterInitializer(nn.init.normal_, mean = init_mean, std = init_std)
        )
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def forward(self, query, key, edge_index, edge_dist):
        """ Provides relative position embeddings for key and query of `num_heads` attention heads. 

        Args:
            query (:obj:`int`): Length of query or query tensor.  
            key (:obj:`int`): Length of key or key tenser.
        Return:
            :obj:`torch.Tensor` of shape ``(num_heads, query_len, key_len)``: Relative position embedding.
        """

        part_buckets = self.num_buckets // (2 if self.bidirectional else 1)
        exact_buckets = part_buckets // 2
        log_buckets = part_buckets - exact_buckets

        if isinstance(query, int):
            query = torch.arange(query, dtype=torch.long, device="cuda")
        if isinstance(key, int):
            key = torch.arange(key, dtype=torch.long, device="cuda")

        if query.dim() == 1:
            relative_position = query[:, None] - key[None, :]
        else:
            relative_position = query[:, :, None] - key[:, None, :]

        neg_pos = relative_position < 0
        relative_position = relative_position.abs()

        small_pos = relative_position < exact_buckets

        log_pos = (torch.clamp(
            torch.log(relative_position.float() / exact_buckets) / math.log(self.max_distance / exact_buckets),
            0,
            0.9999
        ) * log_buckets).long() + exact_buckets

        buckets = torch.where(small_pos, relative_position, log_pos)
        if self.bidirectional:
            buckets = torch.where(
                neg_pos,
                buckets + part_buckets,
                buckets
            )
        else:
            buckets = torch.masked_fill(
                buckets,
                neg_pos,
                0,
            )
        if query.dim() == 1:
            return F.embedding(buckets, self.relative_attention_bias, padding_idx = -1).permute(2, 0, 1).contiguous()
        else:
            return F.embedding(buckets, self.relative_attention_bias, padding_idx = -1).permute(-1, -3, -2).contiguous()
