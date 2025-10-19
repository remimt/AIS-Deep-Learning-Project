import torch
import torch.nn as nn

class MLPtest(nn.Module):
  def __init__(self, c_max, in_dim=5, hidden1=16, hidden2=8, out_dim=2):
    super(MLPtest, self).__init__()
    self.c_max = c_max
    self.layers = nn.Sequential(
        nn.Linear(in_dim, hidden1, bias=True),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2, bias=True),
        nn.ReLU(),
        nn.Linear(hidden2, out_dim, bias=True)
    )
  def forward(self, x_heading, x_cog, x_sog):
    x = torch.stack([torch.sin(x_heading), torch.cos(x_heading),
                   torch.sin(x_cog), torch.cos(x_cog),
                   x_sog],
                  dim=1)
    x_out = self.layers(x)
    return torch.tanh(x_out)*self.c_max

class MLPExt_network(nn.Module):
    def __init__(self, c_max, in_dim=6, hidden1=128, hidden2=32, out_dim=2):
        super(MLPExt_network, self).__init__()
        self.c_max = c_max

        # 🔹 Paramètre apprenable STW
        self.x_stw_nn = nn.Parameter(torch.tensor(1.0))

        # 🔹 Architecture du réseau
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden1, bias=True),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2, bias=True),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim, bias=True)
        )

    def forward(self, x_heading, x_cog, x_sog):
        # Construction du vecteur d'entrée
        batch_size = x_heading.shape[0]
        x_stw_nn = self.x_stw_nn.expand(batch_size)

        x = torch.stack([
            torch.sin(x_cog),
            torch.cos(x_cog),
            x_sog,
            torch.sin(x_heading),
            torch.cos(x_heading),
            x_stw_nn
        ], dim=1)

        x_out = self.layers(x)
        return torch.tanh(x_out) * self.c_max
