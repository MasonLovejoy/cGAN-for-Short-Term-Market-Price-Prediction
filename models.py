from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

PROJECT_DIR = Path(__file__).resolve().parent

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        batch_std = torch.std(x, dim=0, keepdim=True)
        mean_std = torch.mean(batch_std)
        shape = list(x.shape)
        shape[1] = 1
        vals = mean_std.repeat(shape)
        return torch.cat([x, vals], dim=1)
    
class EMAModel:
    """
    Exponential Moving Average of model parameters.
    Improves generator stability and quality.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
class DiscriminatorPlanner(nn.Module):
    def __init__(self, features, filter_num=64, use_minibatch_stddev=True):
        super().__init__()
        from torch.nn.utils import spectral_norm
        
        self.use_minibatch_stddev = use_minibatch_stddev
        
        if use_minibatch_stddev:
            self.minibatch_stddev = MinibatchStdDev()
            
        self.conv_layers = nn.Sequential(
            # Input: (batch, 1, window_size, features)
            spectral_norm(nn.Conv2d(1, filter_num, kernel_size=(3, features), padding=(1, 0))),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            spectral_norm(nn.Conv2d(filter_num, filter_num*2, kernel_size=(3, 1), padding=(1, 0))),
            nn.InstanceNorm2d(filter_num*2),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            
            spectral_norm(nn.Conv2d(filter_num*2, filter_num*4, kernel_size=(3, 1), padding=(1, 0))),
            nn.InstanceNorm2d(filter_num*4),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(filter_num*4, 128)),
            nn.LeakyReLU(0.2)
        )
        
        self.classifier = spectral_norm(nn.Linear(128, 1))
        
    def forward(self, x, return_features=False):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = self.flatten(x)
        features = self.feature_extractor(x)
        output = self.classifier(features)
        
        if return_features:
            return output, features
        return output


class GeneratorPlanner(nn.Module):
    def __init__(self, feature_size, hidden_dim, window_size, target_size):
        super().__init__()
        self.q = target_size
        self.p = window_size
        self.feats = feature_size
        self.hid_dim = hidden_dim
        
        self.encoder = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        
        self.decoder = nn.LSTM(
            input_size=feature_size + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, feature_size),
            nn.Tanh() 
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        _, (h_n, c_n) = self.encoder(x)
        
        noise = torch.randn(batch_size, self.q, self.feats, device=x.device)
        context = h_n[-1].unsqueeze(1).repeat(1, self.q, 1)

        decoder_input = torch.cat([noise, context], dim=-1)
        lstm_out, _ = self.decoder(decoder_input, (h_n, c_n))

        output = self.output_net(lstm_out)
        return output

MODEL_REGISTRY = {
    "discriminator_planner": DiscriminatorPlanner,
    "generator_planner": GeneratorPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Loads a planner model by name with optional pre-trained weights.
    
    Args:
        model_name (str): name of the model architecture
        with_weights (bool): whether to load pre-trained weights
        **model_kwargs: additional model configuration parameters
        
    Returns:
        torch.nn.Module: initialized model
    """
    m = MODEL_REGISTRY[model_name](**model_kwargs)

    if with_weights:
        model_path = PROJECT_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e
    return m

def save_model(model: torch.nn.Module) -> str:
    """
    Saves model state dict to disk.
    
    Args:
        model (torch.nn.Module): model to save
        
    Returns:
        str: path to saved model
    """
    model_name = None

    for n, m in MODEL_REGISTRY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = PROJECT_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path