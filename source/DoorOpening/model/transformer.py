import torch
import torch.nn as nn
from collections import OrderedDict
from collections.abc import Mapping

try:
    from DoorOpening.model.base_model import BaseModel
except ImportError:
    from .base_model import BaseModel
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
from pointnet2_ops.pointnet2_modules import PointnetSAModule


def sample_points_fps(points, features, npoint):
    """
    Sample points from the point cloud using the farthest point sampling algorithm.
    Args:
        points: (B, N, D)
        features: (B, N, F)
        npoint: int
    Returns:
        (B, npoint, D)
    """
    assert points.shape[:2] == features.shape[:2], "Points and features must have the same batch size and number of points"
    features_flipped = features.transpose(1, 2).contiguous()
    npoint = min(npoint, features.shape[1])
    new_features = gather_operation(
        features_flipped, furthest_point_sample(points, npoint)
    ).transpose(1, 2).contiguous()
    return new_features


def strip_prefix_from_state_dict(state_dict, prefix=None):
    """
    Strips a common prefix from all state dict keys that might be added during torch.compile.
    
    Args:
        state_dict: The model state dictionary
        prefix: Optional specific prefix to strip. If None, detects from common prefixes.
    
    Returns:
        State dict with prefix removed from keys
    """
    if prefix is None:
        common_prefixes = ["module._orig_mod.", "module.", "_orig_mod."]
        for p in common_prefixes:
            if any(k.startswith(p) for k in state_dict.keys()):
                prefix = p
                break
    if prefix:
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return state_dict


class PointNetEncoder(nn.Module):
    def __init__(self, output_dim, num_output_tokens, dropout=0):
        super().__init__()

        self.SA_module = PointnetSAModule(
            npoint=num_output_tokens,
            radius=0.1,
            nsample=64,
            mlp=[3, 64, 64, 64],
            bn=False,
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64, output_dim*2),
            nn.LayerNorm(output_dim*2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(output_dim*2, output_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = xyz.clone().transpose(1, 2).contiguous()
        xyz, features = self.SA_module(xyz, features)
        features = features.transpose(1, 2).contiguous()
        return self.fc_layer(features)

class MLPEncoder(nn.Module):
    def __init__(self, output_dim, num_output_tokens, dropout=0):
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Linear(3, output_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim*2, output_dim)
        )

    def forward(self, points):
        if points.shape[1] > self.num_output_tokens:
            points = sample_points_fps(points, points, self.num_output_tokens)
        features = self.net(points)
        return features

class StateEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        dropout=0.1,
        activation="relu",
        use_layer_norm=False,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            dim = int(dim)
            layers.extend([
                nn.Linear(prev_dim, dim),
            ])
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.extend([
                _make_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def _make_activation(name):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation '{name}'.")


class SharedTemporalProprioEncoder(nn.Module):
    def __init__(
        self,
        field_dims,
        timestamps_ms,
        hidden_dims,
        output_dim,
        activation="elu",
        dropout=0.0,
        use_layer_norm=True,
        add_time_embedding=True,
    ):
        super().__init__()
        self.field_dims = OrderedDict((str(key), int(value)) for key, value in field_dims.items())
        self.field_names = tuple(self.field_dims.keys())
        self.timestamps_ms = tuple(int(timestamp) for timestamp in timestamps_ms)
        self.num_timestamps = len(self.timestamps_ms)
        self.input_dim = sum(self.field_dims.values())
        self.output_dim = int(output_dim)
        self.add_time_embedding = bool(add_time_embedding)
        self.use_layer_norm = bool(use_layer_norm)
        self.net = StateEncoder(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=self.output_dim,
            dropout=dropout,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        self.time_embedding = nn.Embedding(self.num_timestamps, self.output_dim) if self.add_time_embedding else None

    def _concatenate_fields(self, proprio_inputs):
        if isinstance(proprio_inputs, torch.Tensor):
            if proprio_inputs.ndim != 3:
                raise RuntimeError(
                    "Expected proprio_temporal tensor with shape [B, T, D], "
                    f"got {tuple(proprio_inputs.shape)}."
                )
            return proprio_inputs
        if not isinstance(proprio_inputs, Mapping):
            raise RuntimeError(
                "proprio_temporal input must be a tensor or mapping of field tensors, "
                f"got {type(proprio_inputs).__name__}."
            )

        batch_size = None
        num_timestamps = None
        parts = []
        for field_name, expected_dim in self.field_dims.items():
            if field_name not in proprio_inputs:
                raise RuntimeError(
                    f"Missing proprio_temporal field '{field_name}'. "
                    f"Expected fields: {list(self.field_names)}."
                )
            value = proprio_inputs[field_name]
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise RuntimeError(
                    f"Expected proprio_temporal['{field_name}'] to have shape [B, T, {expected_dim}], "
                    f"got {tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__}."
                )
            if batch_size is None:
                batch_size = value.shape[0]
                num_timestamps = value.shape[1]
            elif value.shape[:2] != (batch_size, num_timestamps):
                raise RuntimeError(
                    f"Inconsistent batch/timestamp dimensions for proprio_temporal['{field_name}']: "
                    f"expected {(batch_size, num_timestamps)}, got {tuple(value.shape[:2])}."
                )
            if value.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"Unexpected feature dimension for proprio_temporal['{field_name}']: "
                    f"expected {expected_dim}, got {value.shape[-1]}."
                )
            parts.append(value)
        return torch.cat(parts, dim=-1)

    def forward(self, proprio_inputs):
        proprio_tensor = self._concatenate_fields(proprio_inputs)
        batch_size, num_timestamps, input_dim = proprio_tensor.shape
        if num_timestamps != self.num_timestamps:
            raise RuntimeError(
                f"Expected {self.num_timestamps} proprio timestamps, got {num_timestamps}."
            )
        if input_dim != self.input_dim:
            raise RuntimeError(
                f"Expected proprio_temporal feature dim {self.input_dim}, got {input_dim}."
            )

        tokens = self.net(proprio_tensor.reshape(batch_size * num_timestamps, input_dim))
        tokens = tokens.view(batch_size, num_timestamps, self.output_dim)
        if self.time_embedding is not None:
            time_ids = torch.arange(self.num_timestamps, device=tokens.device)
            tokens = tokens + self.time_embedding(time_ids).unsqueeze(0)
        return tokens

class PCDTransformer(BaseModel):
    def __init__(
        self,
        hidden_dim=256,
        type_dim=4,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        chunk_size=1,
        pcd_encoders_cfg=None,
        state_encoders_cfg=None,
        transformer_cfg=None,
        normalize_state=False,
        normalize_action=False,
        action_std=None,
        action_space="delta",
        action_dim=22,
        aux_weight=0.0,
        aux_prediction_mode="absolute",
        aux_delta_scale=0.01,
        mode_prediction=None,
        force_prediction=None,
        proprio_temporal_encoder=None,
    ):
        super().__init__(
            normalize_state=normalize_state,
            normalize_action=normalize_action,
            action_std=action_std,
            action_space=action_space,
        )
        self.hidden_dim = hidden_dim
        self.type_dim = type_dim
        assert self.hidden_dim > self.type_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.pcd_encoders_cfg = pcd_encoders_cfg
        self.state_encoders_cfg = state_encoders_cfg
        self.transformer_cfg = transformer_cfg
        self.proprio_temporal_cfg = dict(proprio_temporal_encoder or {})
        self.proprio_temporal_enabled = bool(self.proprio_temporal_cfg.get("enabled", False))
        self.proprio_temporal_obs_key = "proprio_temporal"
        self.proprio_temporal_encoder = None
        self.proprio_temporal_fields = tuple()
        self.proprio_temporal_timestamps_ms = tuple()
        self.proprio_temporal_field_state_keys = OrderedDict()
        self.proprio_temporal_field_dims = OrderedDict()
        self.proprio_temporal_covered_state_keys = frozenset()
        self.mode_prediction_cfg = mode_prediction or {}
        self.mode_prediction_enabled = bool(self.mode_prediction_cfg.get("enabled", False))
        self.num_modes = int(self.mode_prediction_cfg.get("num_modes", 4))
        self.mode_weight = float(self.mode_prediction_cfg.get("weight", 0.0))
        self.return_latent = bool(self.mode_prediction_cfg.get("return_latent", False))
        self.force_prediction_cfg = force_prediction or {}
        self.force_prediction_enabled = bool(self.force_prediction_cfg.get("enabled", False))
        self.force_output_dim = int(self.force_prediction_cfg.get("output_dim", 3))
        self.force_prediction_weight = float(self.force_prediction_cfg.get("weight", 0.0))
        if self.force_prediction_enabled and self.force_output_dim <= 0:
            raise ValueError("force_prediction.output_dim must be positive when force prediction is enabled.")

        # update config for auxiliary object state prediction
        self.aux_prediction = (aux_weight > 0)
        self.aux_weight = aux_weight
        self.aux_state_keys = [
            key
            for key, cfg in state_encoders_cfg.items()
            if cfg.get("use_state", False) and str(key).startswith("aux_")
        ]
        self.aux_output_dim = sum(int(state_encoders_cfg[key]["input_dim"]) for key in self.aux_state_keys)
        self.aux_memory_allowlist = ["local_pcd_t", *self.aux_state_keys]
        self.aux_prediction_mode = str(aux_prediction_mode).lower()
        if self.aux_prediction_mode not in ["absolute", "delta"]:
            raise ValueError(f"aux_prediction_mode must be 'absolute' or 'delta', got {aux_prediction_mode}")
        self.aux_delta_scale = float(aux_delta_scale)
        if self.aux_prediction and self.aux_output_dim <= 0:
            raise ValueError("aux_weight > 0 requires at least one enabled aux_* state encoder.")

        # Type embeddings and encodersfor different modalities
        self.type_embeddings = nn.ParameterDict()
        self.encoders = nn.ModuleDict()
        self.obs_encoder_order = []
        if self.proprio_temporal_enabled:
            self._initialize_proprio_temporal_encoder()
        for key, cfg in pcd_encoders_cfg.items():
            if cfg["use_pcd"]:
                self.type_embeddings[key] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, type_dim)))
                self.encoders[key] = self._initialize_pcd_encoder(self.hidden_dim-self.type_dim, cfg)
                self.obs_encoder_order.append(key)
        proprio_temporal_inserted = False
        for key, cfg in state_encoders_cfg.items():
            if not cfg["use_state"]:
                continue
            if self.proprio_temporal_enabled and key in self.proprio_temporal_covered_state_keys:
                if not proprio_temporal_inserted:
                    self.obs_encoder_order.append(self.proprio_temporal_obs_key)
                    proprio_temporal_inserted = True
                continue
            self.type_embeddings[key] = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, type_dim)))
            self.encoders[key] = StateEncoder(cfg["input_dim"], cfg["hidden_dims"], self.hidden_dim-self.type_dim, cfg["dropout"])
            self.obs_encoder_order.append(key)
        if self.proprio_temporal_enabled and not proprio_temporal_inserted:
            self.obs_encoder_order.append(self.proprio_temporal_obs_key)

        # Query tokens
        self.aux_query_idx = 0 if self.aux_prediction else None
        next_query_idx = 1 if self.aux_prediction else 0
        self.action_query_start = next_query_idx
        self.action_query_end = self.action_query_start + chunk_size
        next_query_idx = self.action_query_end
        self.mode_query_idx = next_query_idx if self.mode_prediction_enabled else None
        if self.mode_prediction_enabled:
            next_query_idx += 1
        self.force_query_idx = next_query_idx if self.force_prediction_enabled else None
        if self.force_prediction_enabled:
            next_query_idx += 1
        num_query_tokens = next_query_idx
        self.query_tokens = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(num_query_tokens, hidden_dim)))

        # Transformer
        if transformer_cfg["type"] == "encoder_decoder":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_cfg["encoder_heads"],
                dim_feedforward=transformer_cfg["encoder_dim_feedforward"],
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_cfg["encoder_layers"])

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=transformer_cfg["decoder_heads"],
                dim_feedforward=transformer_cfg["decoder_dim_feedforward"],
                dropout=dropout,
                batch_first=True
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=transformer_cfg["decoder_layers"])
        else:
            raise NotImplementedError(f"Transformer type {transformer_cfg['type']} not implemented")

        # Output head
        self.action_head = nn.Linear(hidden_dim, action_dim)  # eef (6) + hand (16)
        if self.aux_prediction:
            self.aux_head = nn.Linear(hidden_dim, self.aux_output_dim)
        if self.mode_prediction_enabled:
            self.mode_head = nn.Linear(hidden_dim, self.num_modes)
        if self.force_prediction_enabled:
            self.force_head = nn.Linear(hidden_dim, self.force_output_dim)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # Note: deal with torch.compile and DDP
        model_state_dict = strip_prefix_from_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            epoch = checkpoint["epoch"]
        else:
            epoch = checkpoint["episode"]
        eval_success_rate, val_loss = None, None
        if "eval_success_rate" in checkpoint:
            eval_success_rate = checkpoint["eval_success_rate"]
        if "val_loss" in checkpoint:
            val_loss = checkpoint["val_loss"]

        self.load_state_dict(model_state_dict)
        return epoch, eval_success_rate, val_loss

    def _initialize_pcd_encoder(self, hidden_dim, encoder_cfg):
        if encoder_cfg["type"] == "mlp":
            return MLPEncoder(hidden_dim, encoder_cfg["num_output_tokens"])
        elif encoder_cfg["type"] == "pointnet":
            return PointNetEncoder(hidden_dim, encoder_cfg["num_output_tokens"])
        else:
            raise ValueError(f"Unknown encoder type: {encoder_cfg['type']}")

    def _get_proprio_temporal_field_alias_candidates(self, field_name):
        alias_map = {
            "tracking_err_arm": ("tracking_err_arm", "target_err_arm"),
            "tracking_err_hand": ("tracking_err_hand", "target_err_hand"),
            "target_err_arm": ("target_err_arm", "tracking_err_arm"),
            "target_err_hand": ("target_err_hand", "tracking_err_hand"),
        }
        return alias_map.get(field_name, (field_name,))

    def _resolve_proprio_temporal_state_key(self, field_name):
        for candidate in self._get_proprio_temporal_field_alias_candidates(field_name):
            if candidate in self.state_encoders_cfg:
                return candidate
        raise RuntimeError(
            f"Could not resolve proprio_temporal field '{field_name}' in state_encoders_cfg. "
            f"Tried aliases {list(self._get_proprio_temporal_field_alias_candidates(field_name))}."
        )

    def _format_temporal_state_key(self, base_key, timestamp_ms):
        timestamp_ms = int(timestamp_ms)
        if timestamp_ms == 0:
            return str(base_key)
        return f"{base_key}_{timestamp_ms}ms"

    def _initialize_proprio_temporal_encoder(self):
        encoder_type = str(self.proprio_temporal_cfg.get("type", "shared_mlp_time_embedding")).lower()
        if encoder_type != "shared_mlp_time_embedding":
            raise ValueError(
                "proprio_temporal_encoder.type must be 'shared_mlp_time_embedding', "
                f"got '{encoder_type}'."
            )
        timestamps_ms = tuple(int(timestamp) for timestamp in self.proprio_temporal_cfg.get("timestamps_ms", []))
        if not timestamps_ms:
            raise ValueError("proprio_temporal_encoder.timestamps_ms must be non-empty when enabled.")
        fields = tuple(str(field) for field in self.proprio_temporal_cfg.get("fields", []))
        if not fields:
            raise ValueError("proprio_temporal_encoder.fields must be non-empty when enabled.")
        hidden_dims = self.proprio_temporal_cfg.get("hidden_dims", [])
        output_dim = int(self.proprio_temporal_cfg.get("output_dim", self.hidden_dim))
        if output_dim != self.hidden_dim:
            raise ValueError(
                "proprio_temporal_encoder.output_dim must match model hidden_dim when the shared encoder is enabled: "
                f"{output_dim} != {self.hidden_dim}."
            )

        field_state_keys = OrderedDict()
        field_dims = OrderedDict()
        covered_state_keys = []
        for field_name in fields:
            state_key = self._resolve_proprio_temporal_state_key(field_name)
            if state_key in field_state_keys.values():
                raise ValueError(
                    f"Duplicate proprio_temporal field mapping for '{field_name}' -> '{state_key}'. "
                    "Each configured field must map to a distinct base state key."
                )
            field_state_keys[field_name] = state_key
            field_dims[field_name] = int(self.state_encoders_cfg[state_key]["input_dim"])
            for timestamp_ms in timestamps_ms:
                covered_state_keys.append(self._format_temporal_state_key(state_key, timestamp_ms))

        self.proprio_temporal_fields = fields
        self.proprio_temporal_timestamps_ms = timestamps_ms
        self.proprio_temporal_field_state_keys = field_state_keys
        self.proprio_temporal_field_dims = field_dims
        self.proprio_temporal_covered_state_keys = frozenset(covered_state_keys)
        self.proprio_temporal_encoder = SharedTemporalProprioEncoder(
            field_dims=field_dims,
            timestamps_ms=timestamps_ms,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=self.proprio_temporal_cfg.get("activation", "elu"),
            dropout=float(self.proprio_temporal_cfg.get("dropout", 0.0)),
            use_layer_norm=bool(self.proprio_temporal_cfg.get("use_layer_norm", True)),
            add_time_embedding=bool(self.proprio_temporal_cfg.get("add_time_embedding", True)),
        )

    def _add_type_embeddings(self, tokens, token_type):
        B = tokens.shape[0]
        type_emb = self.type_embeddings[token_type].expand(B, tokens.shape[1], -1)
        return torch.cat([tokens, type_emb], dim=-1)

    def _postprocess_aux_prediction(self, aux_pred):
        if self.aux_prediction_mode == "delta":
            return torch.clamp(aux_pred, -1.0, 1.0)
        return aux_pred

    def _build_decoder_masks(self, query_tokens, token_ranges):
        num_queries = query_tokens.shape[1]
        num_memory_tokens = 0
        for token_slice in token_ranges.values():
            num_memory_tokens = max(num_memory_tokens, token_slice.stop)

        tgt_mask = torch.zeros((num_queries, num_queries), dtype=torch.bool, device=query_tokens.device)
        memory_mask = torch.zeros((num_queries, num_memory_tokens), dtype=torch.bool, device=query_tokens.device)

        if self.aux_prediction:
            # Aux query is index 0; action queries are index [action_query_start:action_query_end].
            # For cross-attention, allow aux query to read only from the allowlist.
            memory_mask[self.aux_query_idx, :] = True
            for key in self.aux_memory_allowlist:
                if key in token_ranges:
                    memory_mask[self.aux_query_idx, token_ranges[key]] = False

            # For decoder self-attention, block aux query from reading non-aux queries.
            if num_queries > 1:
                tgt_mask[self.aux_query_idx, :] = True
                tgt_mask[self.aux_query_idx, self.aux_query_idx] = False

        if self.mode_prediction_enabled:
            # Keep mode prediction as a readout branch: action queries cannot read the mode query,
            # and the mode query reads encoder memory directly instead of action-query outputs.
            tgt_mask[self.action_query_start:self.action_query_end, self.mode_query_idx] = True
            tgt_mask[self.mode_query_idx, :] = True
            tgt_mask[self.mode_query_idx, self.mode_query_idx] = False

        if self.force_prediction_enabled:
            # Force prediction is also a readout-only branch with no influence on action decoding.
            tgt_mask[self.action_query_start:self.action_query_end, self.force_query_idx] = True
            tgt_mask[self.force_query_idx, :] = True
            tgt_mask[self.force_query_idx, self.force_query_idx] = False

        return tgt_mask, memory_mask

    def _infer_batch_size(self, obs_dict):
        for value in obs_dict.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
            if isinstance(value, Mapping):
                return self._infer_batch_size(value)
        raise RuntimeError("Could not infer batch size from observation dictionary.")

    def forward(self, obs, target=None, action_chunk_idx=None):
        # Get inputs
        obs_dict = obs
        B = self._infer_batch_size(obs_dict)

        obs_tokens = []
        token_ranges = {}
        token_start_idx = 0

        for key in self.obs_encoder_order:
            if key == self.proprio_temporal_obs_key:
                if not self.proprio_temporal_enabled or self.proprio_temporal_encoder is None:
                    raise RuntimeError("proprio_temporal token was requested but the encoder is not initialized.")
                if key not in obs_dict:
                    raise RuntimeError(
                        f"Observation is missing '{key}' while proprio_temporal_encoder is enabled."
                    )
                tokens = self.proprio_temporal_encoder(obs_dict[key])
                typed_tokens = tokens
            else:
                if key not in obs_dict:
                    raise RuntimeError(f"Observation is missing '{key}'.")
                tokens = self.encoders[key](obs_dict[key])
                if len(tokens.shape) == 2:
                    tokens = tokens.unsqueeze(1)
                typed_tokens = self._add_type_embeddings(tokens, key)
            # print(key, tokens.shape)
            # print(typed_tokens.shape)
            obs_tokens.append(typed_tokens)
            token_ranges[key] = slice(token_start_idx, token_start_idx + typed_tokens.shape[1])
            token_start_idx += typed_tokens.shape[1]
        obs_tokens = torch.cat(obs_tokens, dim=1)  # (B, N, H)

        memory = self.encoder(obs_tokens)  # (B, N, H)
        query_tokens = self.query_tokens.expand(B, -1, -1)  # (B, chunk_size/C+1, H)
        if self.aux_prediction or self.mode_prediction_enabled or self.force_prediction_enabled:
            tgt_mask, memory_mask = self._build_decoder_masks(query_tokens, token_ranges)
            output = self.decoder(
                query_tokens,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )  # (B, chunk_size/C+1, H)
        else:
            output = self.decoder(query_tokens, memory)  # (B, chunk_size/C+1, H)

        pred = {}
        if self.return_latent:
            z_context = memory.mean(dim=1)  # (B, H)
            pred["latent"] = z_context
        if self.mode_prediction_enabled:
            output_mode = output[:, self.mode_query_idx, :]  # (B, H)
            pred["mode_logits"] = self.mode_head(output_mode)
        if self.force_prediction_enabled:
            output_force = output[:, self.force_query_idx, :]  # (B, H)
            pred["force"] = self.force_head(output_force)
        if self.aux_prediction:
            output_action = output[:, self.action_query_start:self.action_query_end, :]  # (B, chunk_size, H)
            output_aux = output[:, self.aux_query_idx:self.aux_query_idx+1, :]  # (B, 1, H)
            pred["aux"] = self._postprocess_aux_prediction(self.aux_head(output_aux))  # (B, 1, aux_output_dim)
        else:
            output_action = output[:, self.action_query_start:self.action_query_end, :]  # (B, chunk_size, H)

        pred["action"] = self.action_head(output_action)  # (B, chunk_size, 7)

        # If target is provided, compute loss and return it
        if target is not None:
            loss = self.compute_loss(pred, target, action_chunk_idx)
            return loss

        return pred

    def compute_loss(self, pred, target, action_chunk_idx=None):
        assert len(target.shape) == 3

        loss = {}

        if self.aux_prediction:
            loss["action"] = torch.nn.functional.mse_loss(pred["action"], target[..., :-self.aux_output_dim])
            loss["aux"] = torch.nn.functional.mse_loss(pred["aux"], target[..., -self.aux_output_dim:])
            loss["total"] = loss["action"] + self.aux_weight * loss["aux"]
        else:
            loss["total"] = torch.nn.functional.mse_loss(pred["action"], target)

        return loss

    def forward_pass(self, obs, target):
        # This method is kept for backward compatibility
        # but now just calls forward with the target
        return self.forward(obs, target)

    # Note this function is outdated, maybe update this later, coordinate with the inference scripts
    def get_action(self, obs):
        self.eval()
        current_angles = obs["current_angles"].clone()
        with torch.no_grad():
            pred = self.forward(obs)
        actions = self.decode_actions(current_angles, pred)
        self.train()
        return actions
