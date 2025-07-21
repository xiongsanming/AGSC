import torch
import torch.nn as nn

class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else output_dim
            layers.append(nn.Linear(in_dim, output_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LoRACLIP(nn.Module):
    def __init__(self, lora_vision_model,
                 n_intermediate_layers_to_use,
                 proj_dim, n_proj_layers, dropout_rate):
        super().__init__()
        self.lora_clip_vision_model = lora_vision_model
        try:
            self.base_config = self.lora_clip_vision_model.get_base_model().config
        except AttributeError:
            raise AttributeError("Unable to find configuration object (config) from provided lora_vision_model.")
        
        self.clip_hidden_dim = self.base_config.hidden_size
        self.total_clip_layers = self.base_config.num_hidden_layers
        
        if n_intermediate_layers_to_use is None or n_intermediate_layers_to_use > self.total_clip_layers:
            self.n_intermediate_layers = self.total_clip_layers
        else:
            self.n_intermediate_layers = n_intermediate_layers_to_use
            
        self.q1 = ProjectionNetwork(self.clip_hidden_dim, proj_dim, n_proj_layers, dropout_rate)
        
        self.tie_weights = nn.Parameter(torch.empty(self.n_intermediate_layers, 1))
            
        self.q2 = ProjectionNetwork(proj_dim, proj_dim, n_proj_layers, dropout_rate)
        
        self.classification_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(proj_dim, 1))

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        expected_dtype = next(self.lora_clip_vision_model.parameters()).dtype
        pixel_values_converted = pixel_values.to(dtype=expected_dtype)
        
        outputs = self.lora_clip_vision_model(
            pixel_values=pixel_values_converted, output_hidden_states=True, return_dict=True
        )
        
        all_hidden_states = outputs.hidden_states
          
        k_tilde = None
        start_idx = (self.total_clip_layers + 1) - self.n_intermediate_layers
        intermediate_cls_tokens = [all_hidden_states[i][:, 0, :] for i in range(start_idx, self.total_clip_layers + 1)]
        k = torch.stack(intermediate_cls_tokens, dim=1)
        k_fp32 = k.float()
        
        k_reshaped = k_fp32.reshape(-1, k_fp32.shape[-1])
        kq1_reshaped = self.q1(k_reshaped)
        kq1 = kq1_reshaped.view(batch_size, self.n_intermediate_layers, -1)
        importance_scores = torch.softmax(self.tie_weights.float(), dim=0)
        importance_scores_batch = importance_scores.unsqueeze(0)
        weighted_kq1 = kq1 * importance_scores_batch.to(kq1.dtype)
        k_tilde = torch.sum(weighted_kq1, dim=1)
            
        kq2 = self.q2(k_tilde)
        logits = self.classification_head(kq2)
        return logits, kq2