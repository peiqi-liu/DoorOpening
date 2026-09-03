"""Capture and interpret decoder cross-attention from a PCDTransformer for visualization.

The PCDTransformer decodes an ACTION query (plus optional aux/mode/door queries) that
cross-attends to the encoder memory. The memory tokens ARE the observation modalities
(point-cloud tokens, temporal proprio, push/pull & left/right conditions, aux anchor, ...),
and ``PCDTransformer.last_token_ranges`` maps memory columns -> obs keys. This module hooks
the decoder's cross-attention, forces PyTorch to return the (otherwise discarded) weights,
and aggregates the ACTION query's attention into:

  - per-modality attention mass (how much the action attends to each obs key), and
  - per-point attention over the point cloud's FPS token centers (a spatial heatmap).

PyTorch's ``TransformerDecoderLayer`` calls ``multihead_attn(..., need_weights=False)`` and
takes ``[0]``, so the weights are normally never computed. The forward-pre-hook here flips
``need_weights=True`` (and ``average_attn_weights=True``) so the forward-hook can read them.
"""

import torch


class AttentionCapture:
    """Hook a ``PCDTransformer`` decoder and read the action-query cross-attention per forward.

    Example::

        cap = AttentionCapture(model)     # model: PCDTransformer (unwrap DDP first)
        pred = model(obs)                 # exactly one forward
        info = cap.pop(model, env_id=0)   # attention for env 0 from that forward, or None
        ...
        cap.remove()
    """

    def __init__(self, model):
        self._handles = []
        self._layer_weights = []
        decoder = getattr(model, "decoder", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError(
                "AttentionCapture expects model.decoder.layers (an nn.TransformerDecoder)."
            )
        for layer in layers:
            mha = getattr(layer, "multihead_attn", None)
            if mha is None:
                continue
            self._handles.append(mha.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
            self._handles.append(mha.register_forward_hook(self._post_hook))
        self.num_layers = len(layers)

    @staticmethod
    def _pre_hook(module, args, kwargs):
        # Force the cross-attention to compute + return head-averaged weights.
        kwargs = dict(kwargs)
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = True
        return args, kwargs

    def _post_hook(self, module, args, output):
        # output = (attn_output, attn_weights); attn_weights is [B, Lq, Sk] (head-averaged).
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self._layer_weights.append(output[1].detach())

    def _drain(self):
        weights = self._layer_weights
        self._layer_weights = []
        return weights

    def pop(self, model, env_id=0):
        """Aggregate the attention captured since the last ``pop`` (i.e. from one forward).

        Returns a dict with:
          - ``per_modality``: {obs_key: float}  attention mass on each modality (sums to ~1),
          - ``pcd``: {pcd_key: (xyz[T,3] cpu, attn[T] cpu)}  per-token attention + token centers,
          - ``action_query_attn``: [Sk] cpu  raw per-memory-token attention for the action query.
        Returns ``None`` if no attention was captured (no forward happened).
        """
        layer_weights = self._drain()
        if not layer_weights:
            return None
        # [L, B, Lq, Sk] -> mean over decoder layers -> [B, Lq, Sk]
        attn = torch.stack(layer_weights, dim=0).mean(dim=0)
        qs, qe = int(model.action_query_start), int(model.action_query_end)
        # action queries (chunk) -> mean over the chunk -> [B, Sk]
        action_attn = attn[:, qs:qe, :].mean(dim=1)
        env_id = max(0, min(int(env_id), action_attn.shape[0] - 1))
        row = action_attn[env_id]  # [Sk]

        token_ranges = model.last_token_ranges or {}
        # One human-readable label PER memory token/column (no aggregation): "pointcloud",
        # "q_arm 300ms", "aux_handle_pos", "target_err_arm", "push_pull_cond", ...
        labels = self._build_token_labels(model, token_ranges, int(row.shape[0]))

        pcd = {}
        for key, xyz in model.get_last_pcd_token_xyz().items():
            sl = token_ranges.get(key)
            if sl is None:
                continue
            attn_tok = row[sl]
            if attn_tok.shape[0] != xyz.shape[1]:
                continue  # token-count mismatch guard
            pcd[key] = (xyz[env_id].detach().float().cpu(), attn_tok.detach().float().cpu())

        return {
            "token_labels": labels,                       # list[str], one per memory token
            "token_attn": row.detach().float().cpu(),     # [Sk] attention per memory token
            "pcd": pcd,                                    # {key: (xyz[T,3], attn[T])} for pcd tokens
        }

    @staticmethod
    def _build_token_labels(model, token_ranges, total):
        """One label per memory column, expanding the packed proprio_temporal block into
        per-(field, timestamp) names (e.g. 'q_arm 300ms', 'base_vel 300ms') and aggregating all
        point-cloud tokens under 'pointcloud'. Other single-token modalities keep their obs key."""
        labels = [None] * total
        ptk = getattr(model, "proprio_temporal_obs_key", None)
        fields = list(getattr(model, "temporal_state_fields", ()) or ())
        ts_ms = list(getattr(model, "temporal_state_timestamps_ms", ()) or ())
        pcd_keys = set(getattr(model, "pcd_encoder_keys", ()) or ())
        for key, sl in token_ranges.items():
            n = sl.stop - sl.start
            if key == ptk and fields and ts_ms:
                n_ts, n_f = len(ts_ms), len(fields)
                if n == n_f * n_ts:
                    # TemporalFieldStateEncoder: field-major (field_idx * n_ts + time_idx).
                    for j in range(n):
                        labels[sl.start + j] = f"{fields[j // n_ts]} {ts_ms[j % n_ts]}ms"
                elif n == n_ts:
                    # SharedTemporalProprioEncoder: one token per timestamp (all fields packed).
                    for j in range(n):
                        labels[sl.start + j] = f"proprio {ts_ms[j]}ms"
                else:
                    for j in range(n):
                        labels[sl.start + j] = f"{key}[{j}]"
            elif key in pcd_keys:
                for j in range(n):
                    labels[sl.start + j] = "pointcloud"
            else:
                for j in range(n):
                    labels[sl.start + j] = key if n == 1 else f"{key}[{j}]"
        return labels

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.remove()
        return False


def top_token_str(token_labels, token_attn, top_k=10):
    """Compact 'label=attn' string for the top-attended INDIVIDUAL tokens (no aggregation)."""
    vals = token_attn.tolist()
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:top_k]
    return "  ".join(f"{token_labels[i]}={vals[i]:.3f}" for i in order)
