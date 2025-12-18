from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

PROJECT_DIR = Path(__file__).resolve().parent

class DiscriminatorPlanner(nn.Module):
    def __init__(self, features, filter_num=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Input: (batch, 1, window_size, features)
            spectral_norm(nn.Conv2d(1, filter_num, kernel_size=(3, features), padding=(1, 0))),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            spectral_norm(nn.Conv2d(filter_num, filter_num*2, kernel_size=(3, 1), padding=(1, 0))),
            nn.BatchNorm2d(filter_num*2),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            spectral_norm(nn.Conv2d(filter_num*2, filter_num*4, kernel_size=(3, 1), padding=(1, 0))),
            nn.BatchNorm2d(filter_num*4),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            spectral_norm(nn.Linear(filter_num*4, 1))
        )
        
    def forward(self, x):
        # Don't use sigmoid in discriminator for WGAN
        return self.model(x)


class GeneratorPlanner(nn.Module):
    def __init__(self, feature_size, hidden_dim, window_size, target_size):
        super().__init__()
        self.q = target_size
        self.p = window_size
        self.feats = feature_size
        self.hid_dim = hidden_dim
        
        # Encoder: Process historical data
        self.encoder = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_dim,
            num_layers=2,  # Increased depth
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder: Generate future sequences
        self.decoder = nn.LSTM(
            input_size=feature_size + hidden_dim,  # Concat noise + context
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer with residual connection
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, feature_size)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode historical context
        _, (h_n, c_n) = self.encoder(x)
        
        # Generate noise for each time step
        noise = torch.randn(batch_size, self.q, self.feats, device=x.device)
        
        # Repeat hidden state for each time step
        context = h_n[-1].unsqueeze(1).repeat(1, self.q, 1)
        
        # Concatenate noise with context
        decoder_input = torch.cat([noise, context], dim=-1)
        
        # Decode
        lstm_out, _ = self.decoder(decoder_input, (h_n, c_n))
        
        # Generate output
        output = self.output(lstm_out)
        
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