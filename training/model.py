import torch
import torch.nn as nn

class VelocityMLP(nn.Module):
    """
    Lightweight MLP for pedestrian velocity regression.
    Input:  flattened sliding window of T frames x 4 features
            [rel_x, rel_y, dx, dy] per frame → 40 features at T=10
    Output: [vx, vy] in m/s
    """
    def __init__(self, input_dim=40, hidden_dims=[256, 128, 64], dropout=0.2):
        super(VelocityMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer — no activation, regression output
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    model = VelocityMLP()
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Forward pass test
    dummy = torch.randn(32, 40)
    out   = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")