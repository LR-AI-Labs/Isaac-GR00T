from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from diffusers.models.embeddings import SinusoidalPositionalEmbedding

from .cross_attention_dit import BasicTransformerBlock
from .flow_matching_action_head import CategorySpecificMLP, MultiEmbodimentActionEncoder

class L1ActionGeneratorConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    action_backbone_cfg: dict = field(
        default=None, metadata={"help": "Action backbone configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_action_backbone: bool = field(
        default=True, metadata={"help": "Whether to tune the backbone of action head."}
    )
    num_past_actions: int = field(
        default=0, metadata={"help": "Number of past actions used."}
    )
    regression_loss: str = field(
        default="l2", metadata={"help": "Loss function for action prediction."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
            
class L1ActionBackbone(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = None, # default is None because add_pos_embed is True
        interleave_self_attention=False,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.gradient_checkpointing = False

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        
        print(
            "Total number of action backbone's parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        # Process through transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                )
            all_hidden_states.append(hidden_states)

        if return_all_hidden_states:
            return self.norm_out(hidden_states), all_hidden_states
        else:
            return self.norm_out(hidden_states)
        
    
class L1ActionGenerator(nn.Module):
    config_class = L1ActionGeneratorConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: L1ActionGeneratorConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = L1ActionBackbone(**config.action_backbone_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        # action_encoder for encoding past actions
        self.num_past_actions = config.num_past_actions
        if self.num_past_actions > 0:
            self.action_encoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.action_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
        else:
            self.action_encoder = None
                        
        self.action_embedding = nn.Embedding(
            config.max_num_embodiments,
            self.input_embedding_dim,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.input_embedding_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.action_embedding.weight.data.normal_(mean=0.0, std=0.02)
        if config.add_pos_embed:
            self.position_embedding = SinusoidalPositionalEmbedding(self.input_embedding_dim, config.max_seq_len)
            # self.position_embedding = nn.Embedding(
            #     config.max_seq_len,
            #     self.input_embedding_dim,
            #     _freeze=True
            # )
            # create_sinusoidal_embeddings(config.max_seq_len, self.input_embedding_dim, self.position_embedding.weight)
        self.regression_loss = config.regression_loss
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_action_backbone)

    def set_trainable_parameters(self, tune_projector: bool, tune_action_backbone: bool):
        self.tune_projector = tune_projector
        self.tune_action_backbone = tune_action_backbone
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_action_backbone:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head backbone: {self.tune_action_backbone}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_action_backbone:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_embedding.eval()
                self.action_decoder.eval()
                if self.action_encoder is not None:
                    self.action_encoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_action_backbone:
                self.model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        
        # Embed action placeholder tokens.
        action_tokens = embodiment_id.unsqueeze(1).repeat(1, self.config.action_horizon).to(device)
        action_features = self.action_embedding(action_tokens)

        # Encode previous actions.
        if self.action_encoder is not None:
            # action_input.action is of shape B x (max_past_action + action_horizon) x action_dim, 
            # just load last "num_past_actions" past actions
            start = -self.action_horizon - self.num_past_actions 
            end = -self.action_horizon
            past_actions = action_input.action[:, start:end, :]
            past_action_features = self.action_encoder(past_actions, embodiment_id)
            action_features = torch.cat((action_features, past_action_features), dim=1)
        
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            action_features = self.position_embedding(action_features)
            # pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            # pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            # action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1) 
        # B x (1+num_past_actions+action_horizon) x input_embedding_dim
        vl_embs = vl_embeds
        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
        )
        pred = self.action_decoder(model_output, embodiment_id) # B x (1+action_horizon) x action_dim
        pred_actions = pred[:, -self.action_horizon:, :] # B x action_horizon x action_dim
        
        # compute regression loss
        action_mask = action_input.action_mask[:, -self.action_horizon:, :]
        ground_truth_actions = action_input.action[:, -self.action_horizon:, :]
        if self.regression_loss == "l2":
            loss = F.mse_loss(pred_actions, ground_truth_actions, reduction="none") * action_mask
        elif self.regression_loss == "l1":
            loss = F.l1_loss(pred_actions, ground_truth_actions, reduction="none") * action_mask
        else:
            raise ValueError(f"Unsupported regression loss: {self.config.regression_loss}")
        loss = loss.sum() / action_mask.sum()
        output_dict = {
            "loss": loss,
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        
        # Embed action placeholder tokens.
        action_tokens = embodiment_id.unsqueeze(1).repeat(1, self.config.action_horizon).to(device)
        action_features = self.action_embedding(action_tokens)

        # Encode previous actions.
        if self.action_encoder is not None:
            # action_input.action is of shape B x (max_past_action + action_horizon) x action_dim, 
            # just load last "num_past_actions" past actions
            start = -self.action_horizon - self.num_past_actions 
            end = -self.action_horizon
            past_actions = action_input.action[:, start:end, :]
            past_action_features = self.action_encoder(past_actions, embodiment_id)
            action_features = torch.cat((action_features, past_action_features), dim=1)
        
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            action_features = self.position_embedding(action_features)
            # pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            # pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            # action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1) 
        # B x (1+num_past_actions+action_horizon) x input_embedding_dim
        vl_embs = vl_embeds
        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
        )
        pred = self.action_decoder(model_output, embodiment_id) # B x (1+action_horizon) x action_dim
        actions = pred[:, -self.action_horizon:, :] # B x action_horizon x action_dim

        return BatchFeature(data={"action_pred": actions})

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()