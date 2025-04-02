import torch

def euclidean_distance(c0, c1):
    """
    Computes the Euclidean distance between two points.

    Parameters:
        c0 (Tensor): Starting point in latent space (shape: (latent_dim,))
        c1 (Tensor): Ending point in latent space (shape: (latent_dim,))

    Returns:
        Tensor: Euclidean distance between c0 and c1.
    """
    return torch.norm(c1 - c0)

def linear_curve(t, c0, c1):
    """
    Constructs a linear curve between two latent points.

    Parameters:
        t (Tensor): 1D tensor of time values (values between 0 and 1)
        c0 (Tensor): Starting point in latent space (shape: (latent_dim,))
        c1 (Tensor): Ending point in latent space (shape: (latent_dim,))

    Returns:
        Tensor: Curve points of shape (len(t), latent_dim)
    """
    return (1 - t).unsqueeze(1) * c0 + t.unsqueeze(1) * c1

def compute_euclidean_path(c0, c1, N=100):
    """
    Computes a straight-line path between two latent points using Euclidean distance.

    Parameters:
        c0 (Tensor): Starting point in latent space (shape: (latent_dim,))
        c1 (Tensor): Ending point in latent space (shape: (latent_dim,))
        N (int): Number of steps in the discretized curve

    Returns:
        Tensor: Euclidean path points of shape (N+1, latent_dim)
    """
    return linear_curve(torch.linspace(0, 1, N + 1), c0, c1)
