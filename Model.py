import torch
import torch.nn.functional as F
from torch import nn
from Params import args
from Utils.Utils import pairPredict
from Transformer import TransformerEncoderLayer


class DualChannelRec(nn.Module):
    def __init__(self):
        super(DualChannelRec, self).__init__()

        # GNN embeddings
        self.user_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.user, args.latdim)))
        self.item_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.item, args.latdim)))

        # ---- Channel 2: Sequence Transformer (skip when gnn_only) ----
        if args.mode != 'gnn_only':
            # Separate item embedding for Transformer to avoid gradient conflict
            # with GNN's item_embeding — each channel optimizes its own embedding space
            self.seq_item_embeding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(args.item, args.latdim)))
            self.pos_embedding = nn.Embedding(args.seq_maxlen, args.latdim)
            self.seq_transformer = TransformerEncoderLayer(
                d_model=args.latdim, num_heads=args.num_head, dropout=args.dropout
            )
            # Pre-compute causal mask (upper-triangular = True means masked)
            causal = torch.triu(torch.ones(args.seq_maxlen, args.seq_maxlen, dtype=torch.bool), diagonal=1)
            self.register_buffer('causal_mask', causal)

        # ---- Gated Fusion (only needed in both mode) ----
        if args.mode == 'both':
            self.gate_linear = nn.Linear(2 * args.latdim, 1)
            # Initialize bias so sigmoid(bias) ≈ 0.88 → heavily favor GNN at start.
            # As Transformer improves, the gate learns to incorporate it.
            nn.init.constant_(self.gate_linear.bias, 2.0)

    # ==================================================================
    # Channel 1: LightGCN (2-layer, no learnable params besides embeddings)
    # ==================================================================
    def gnn_message_passing(self, adj, embeds):
        with torch.amp.autocast('cuda', enabled=False):
            return torch.spmm(adj, embeds.float())

    def lightgcn_forward(self, adj):
        """LightGCN with dual-pass: returns LPF (standard) + HPF user embeds."""
        embeds_0 = torch.cat([self.user_embeding, self.item_embeding], dim=0)
        lpf_list = [embeds_0]   # layer-0 is shared (original embeddings)
        hpf_list = []           # high-pass components per layer

        cur = embeds_0
        for _ in range(args.block_num):
            smoothed = self.gnn_message_passing(adj, cur)
            hpf_list.append(cur - smoothed)   # high-pass: identity − smoothed
            lpf_list.append(smoothed)
            cur = smoothed

        # Standard LightGCN aggregation (sum over all layers including layer-0)
        lpf_embeds = sum(lpf_list)
        # HPF aggregation (sum of per-layer high-pass components)
        hpf_embeds = sum(hpf_list)

        gnn_user_embeds = lpf_embeds[:args.user]
        gnn_item_embeds = lpf_embeds[args.user:]
        hpf_user_embeds = hpf_embeds[:args.user]
        return gnn_user_embeds, gnn_item_embeds, hpf_user_embeds

    # ==================================================================
    # Channel 2: Sequence Transformer
    # ==================================================================
    def _build_combined_mask(self, seq_mask):
        """
        Build a per-sample float attention mask that combines causal + padding,
        guaranteeing no row is ALL -inf (which would cause NaN in softmax).

        For padding rows where all causal keys are also padding, the diagonal
        is opened so the position can attend to itself — producing a well-defined
        (though unused) output with clean gradients for all internal parameters.

        Args:
            seq_mask: (B, S) bool, True = padding position
        Returns:
            combined: (B * num_head, S, S) float attention mask
        """
        B = seq_mask.shape[0]
        S = args.seq_maxlen
        device = seq_mask.device

        # Causal float mask: upper triangle = -inf, lower triangle + diag = 0
        causal_float = torch.where(
            self.causal_mask,
            torch.tensor(float('-inf'), device=device),
            torch.tensor(0.0, device=device),
        )  # (S, S)

        # Expand to (B, S, S) and add padding column mask
        combined = causal_float.unsqueeze(0).expand(B, -1, -1).clone()
        pad_col = seq_mask.unsqueeze(1).expand(-1, S, -1)  # (B, S, S)
        combined[pad_col] = float('-inf')

        # Fix all-masked rows: open diagonal so position self-attends
        all_neg_inf = (combined == float('-inf')).all(dim=-1)  # (B, S)
        if all_neg_inf.any():
            rows, cols = all_neg_inf.nonzero(as_tuple=True)
            combined[rows, cols, cols] = 0.0

        # Reshape for multi-head: (B, S, S) → (B*nhead, S, S)
        combined = combined.unsqueeze(1).expand(-1, args.num_head, -1, -1)
        combined = combined.reshape(B * args.num_head, S, S)
        return combined

    def seq_forward(self, user_seq, seq_mask):
        """
        Causal Transformer over user interaction history.
        Args:
            user_seq:  (batch, seq_maxlen) int64, item ids (0 = padding)
            seq_mask:  (batch, seq_maxlen) bool, True = padding position
        Returns:
            seq_user_embeds: (batch, latdim)
        """
        # Use Transformer's own item embeddings (separate from GNN's)
        item_embs = self.seq_item_embeding[user_seq]  # (batch, seq_maxlen, latdim)

        # Add positional encoding
        positions = torch.arange(args.seq_maxlen, device=user_seq.device)
        item_embs = item_embs + self.pos_embedding(positions).unsqueeze(0)

        # nn.MultiheadAttention expects (seq_len, batch, d_model)
        item_embs = item_embs.transpose(0, 1)  # (seq_maxlen, batch, latdim)

        # Combined causal + padding mask prevents NaN at source:
        # every row has at least one attendable key, so softmax never
        # receives all -inf → no NaN in forward → clean gradients for
        # all internal parameters (W_Q, W_K, W_V, FFN, Norm).
        combined_mask = self._build_combined_mask(seq_mask)

        out = self.seq_transformer(
            item_embs,
            attn_mask=combined_mask,  # (B*nhead, S, S) combined float mask
            length_mask=None          # already incorporated into combined_mask
        )
        out = out.transpose(0, 1)  # (batch, seq_maxlen, latdim)

        # Take the last position (left-padded, so position seq_maxlen-1 is always
        # the most recent item and has the richest causal context)
        seq_user_embeds = out[:, args.seq_maxlen - 1, :]  # (batch, latdim)

        return seq_user_embeds

    # ==================================================================
    # Gated Fusion
    # ==================================================================
    def gated_fusion(self, gnn_user, seq_user):
        """
        Args:
            gnn_user: (batch, latdim)
            seq_user: (batch, latdim)
        Returns:
            fused: (batch, latdim)
        """
        concat = torch.cat([gnn_user, seq_user], dim=-1)  # (batch, 2*latdim)
        alpha = torch.sigmoid(self.gate_linear(concat))     # (batch, 1)
        fused = alpha * gnn_user + (1 - alpha) * seq_user
        return fused

    # ==================================================================
    # Forward
    # ==================================================================
    def forward(self, adj, user_seq, seq_mask, ancs=None):
        """
        Args:
            adj:      normalized bipartite adjacency (sparse)
            user_seq: (batch, seq_maxlen) int64
            seq_mask: (batch, seq_maxlen) bool, True=padding
            ancs:     (batch,) int64, user indices for this batch (training)
                      None means full-user mode (testing, user_seq covers all users)
        Returns:
            final_user_embeds: (batch, latdim) or (num_users, latdim) fused embeddings
            item_embeds:       (num_items, latdim) from GNN
        """
        mode = args.mode

        if mode == 'gnn_only':
            gnn_user_embeds, gnn_item_embeds, hpf_user_embeds = self.lightgcn_forward(adj)
            if ancs is not None:
                return gnn_user_embeds[ancs], gnn_item_embeds, gnn_user_embeds[ancs], hpf_user_embeds[ancs]
            return gnn_user_embeds, gnn_item_embeds

        if mode == 'transformer_only':
            seq_user_embeds = self.seq_forward(user_seq, seq_mask)
            item_embeds = self.item_embeding
            return seq_user_embeds, item_embeds

        # mode == 'both': GNN + Transformer with gated fusion
        gnn_user_embeds, gnn_item_embeds, hpf_user_embeds = self.lightgcn_forward(adj)
        seq_user_embeds = self.seq_forward(user_seq, seq_mask)

        if ancs is not None:
            gnn_user_batch = gnn_user_embeds[ancs]
            hpf_user_batch = hpf_user_embeds[ancs]
            final_user = self.gated_fusion(gnn_user_batch, seq_user_embeds)
        else:
            gnn_user_batch = gnn_user_embeds
            hpf_user_batch = hpf_user_embeds
            final_user = self.gated_fusion(gnn_user_embeds, seq_user_embeds)

        return final_user, gnn_item_embeds, gnn_user_batch, seq_user_embeds, hpf_user_batch

    # ==================================================================
    # Losses
    # ==================================================================
    def bprLoss(self, user_embeds, item_embeds, poss, negs):
        """BPR loss. user_embeds is already fused (batch, latdim)."""
        posEmbeds = item_embeds[poss]
        negEmbeds = item_embeds[negs]
        scoreDiff = pairPredict(user_embeds, posEmbeds, negEmbeds)
        bprLoss = - ((scoreDiff).sigmoid() + 1e-6).log().mean()
        return bprLoss

    def infoNCELoss(self, view1, view2):
        """
        Cross-channel InfoNCE contrastive loss.
        Pulls the same user's GNN and Transformer representations together,
        pushes different users apart.
        Args:
            view1: (batch, latdim) - GNN user embeddings
            view2: (batch, latdim) - Transformer user embeddings
        Returns:
            cl_loss: scalar
        """
        view1 = F.normalize(view1, dim=-1)
        view2 = F.normalize(view2, dim=-1)
        # (batch, batch) similarity matrix
        logits = torch.mm(view1, view2.t()) / args.temp
        labels = torch.arange(view1.shape[0], device=view1.device)
        # Symmetric loss: view1→view2 and view2→view1
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss

    # ==================================================================
    # Predict (test time — all users)
    # ==================================================================
    def predict(self, adj, all_user_seqs, all_seq_masks):
        """
        Full-user prediction for evaluation.
        Args:
            adj:             normalized bipartite adjacency (sparse)
            all_user_seqs:   (num_users, seq_maxlen) int64
            all_seq_masks:   (num_users, seq_maxlen) bool
        Returns:
            user_embeds: (num_users, latdim)
            item_embeds: (num_items, latdim)
        """
        mode = args.mode

        if mode == 'gnn_only':
            gnn_user_embeds, gnn_item_embeds, _ = self.lightgcn_forward(adj)
            return gnn_user_embeds, gnn_item_embeds

        # Compute sequence embeddings (used by transformer_only and both)
        num_users = all_user_seqs.shape[0]
        batch_size = args.tstBat
        seq_embeds_list = []
        for start in range(0, num_users, batch_size):
            end = min(start + batch_size, num_users)
            seq_batch = all_user_seqs[start:end]
            mask_batch = all_seq_masks[start:end]
            seq_embeds_list.append(self.seq_forward(seq_batch, mask_batch))
        seq_user_embeds = torch.cat(seq_embeds_list, dim=0)

        if mode == 'transformer_only':
            return seq_user_embeds, self.item_embeding

        # mode == 'both'
        gnn_user_embeds, gnn_item_embeds, _ = self.lightgcn_forward(adj)
        final_user_embeds = self.gated_fusion(gnn_user_embeds, seq_user_embeds)
        return final_user_embeds, gnn_item_embeds
