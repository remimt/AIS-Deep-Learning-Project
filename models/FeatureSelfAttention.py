import torch
import torch.nn as nn

def ang_diff(cog, heading):
    """Retourn wrap(cog - heading) in (-pi, pi]."""
    s = torch.sin(cog - heading)
    c = torch.cos(cog - heading)
    return torch.atan2(s, c)

class SABlock(nn.Module):
    def __init__(self, d_model=64, n_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        attn_out, attn = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        f = self.ffn(x)
        x = self.norm2(x + self.drop(f))
        if return_attn:
            # attn: [batch, n_heads, F, F]
            return x, attn
        return x, None

class FeatureSelfAttention_network(nn.Module):
    """
    Auto-attention entre N features scalaires.
    - On projette chaque scalaire -> vecteur (projection partagée)
    - On ajoute une 'embedding d'identité' par feature (apprend qui est qui)
    - On passe plusieurs blocs d'auto-attention
    - On pool (moyenne) et on prédit
    """
    def __init__(self, c_max, n_features=10, d_model=64, n_heads=8, n_layers=2, out_dim=2, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.scalar2vec = nn.Linear(1, d_model)                # projection partagée 1->d
        self.feature_id = nn.Parameter(torch.randn(n_features, d_model))  # embedding par feature
        self.blocks = nn.ModuleList([SABlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, out_dim)                 # tête de sortie (régression)
        self.c_max = c_max

    def forward(self, x_heading, x_cog, x_sog, x_length, x_width, x_draft, return_attn=False):
        x_dch = ang_diff(x_cog, x_heading)
        x_trig = torch.stack([torch.sin(x_heading), torch.cos(x_heading),
                              torch.sin(x_cog), torch.cos(x_cog),
                              torch.sin(x_dch), torch.cos(x_dch),
                              x_sog,
                              x_length, x_width, x_draft],
                             dim=1)

        B, F = x_trig.shape
        assert F == self.n_features, f"Attendu {self.n_features} features, reçu {F}"
        # 1) projeter scalaires -> vecteurs
        h = self.scalar2vec(x_trig.unsqueeze(-1))                    # [B, F, d_model]
        # 2) ajouter l'identité de feature
        h = h + self.feature_id.unsqueeze(0)                    # [B, F, d_model]
        # 3) blocs d'auto-attention
        attn_list = []
        for blk in self.blocks:
            h, attn = blk(h, return_attn)
            if return_attn and attn is not None:
                attn_list.append(attn)                          # [B, n_heads, F, F]
        # 4) pooling (moyenne sur les features) puis prédiction
        z = h.mean(dim=1)                                       # [B, d_model]
        y = self.head(z)
        out = torch.tanh(y) * self.c_max                                        # [B, out_dim]
        if return_attn:
            return out, attn_list
        return out