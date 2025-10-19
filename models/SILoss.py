import torch

def si_loss(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    SI = sum_i num_i / sum_i ||x_i||^2

    Règle pour num_i :
      - si (||y_i|| > ||x_i||) et (<y_i, x_i> > 0):
            y_modif_i = (2||x_i|| - ||y_i||) * (y_i / ||y_i||)
            num_i = <y_modif_i, x_i>
        (si 2||x|| - ||y|| < 0, le facteur négatif inverse la direction → pénalisation)
      - sinon:
            num_i = <y_i, x_i>
    """
    eps = torch.finfo(Y.dtype).eps

    # Bases
    dot = (Y * X).sum(dim=1)           # <y_i, x_i>
    ny  = torch.norm(Y, dim=1)         # ||y_i||
    nx  = torch.norm(X, dim=1)         # ||x_i||

    # Condition
    cond = (ny > nx) & (dot > 0)

    # y_unit = y / ||y||
    y_unit = Y / ny.clamp_min(eps).unsqueeze(1)

    # Norme signée: peut être négative → inverse la direction si nécessaire
    new_norm_signed = 2 * nx - ny

    # <y_modif_i, x_i> = (2||x|| - ||y||) * <y_unit, x_i>
    dot_yunit_x = (y_unit * X).sum(dim=1)
    num_mod = new_norm_signed * dot_yunit_x

    # Appliquer la règle
    num_i = torch.where(cond, num_mod, dot)

    # Assemblage
    num = num_i.sum()
    den = (nx**2).sum().clamp_min(torch.finfo(X.dtype).eps)
    return num / den