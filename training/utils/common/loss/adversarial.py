import torch
from torch.nn import functional as F


def dis_loss(
    loss_type: str,
    logits_real: torch.Tensor,
    logits_fake: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the discriminator loss based on the specified loss type.

    Args:
        loss_type: The type of loss function to use.
            Supported values are "hinge", "nonsaturating", and "wgan".
        logits_real: The logit output on real samples.
        logits_fake: The logit output on fake samples.
    Returns:
        The discriminator loss.
    """
    if loss_type == "hinge":
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    elif loss_type == "nonsaturating":
        loss_real = torch.mean(F.softplus(-logits_real))
        loss_fake = torch.mean(F.softplus(logits_fake))
    elif loss_type == "wgan":
        loss_real = -torch.mean(logits_real)
        loss_fake = torch.mean(logits_fake)
    else:
        raise NotImplementedError

    return 0.5 * (loss_real + loss_fake)


def gen_loss(
    loss_type: str,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the generator loss based on the specified loss type.

    Args:
        loss_type: The type of loss function to use.
            Supported values are "hinge", "nonsaturating" and "wgan".
        logits: the logit output on the generated samples.
    Returns:
        The generator loss.
    """
    if loss_type == "hinge":
        loss = -torch.mean(logits)
    elif loss_type == "nonsaturating":
        loss = torch.mean(F.softplus(-logits))
    elif loss_type == "wgan":
        loss = -torch.mean(logits)
    else:
        raise NotImplementedError

    return loss


def r1_regularization(
    inputs: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    R1 regularization (https://arxiv.org/pdf/1801.04406, eq.9)

    Args:
        inputs: The input tensor to the discriminator. requires_grad must be True.
        logits: The output logits of the discriminator.
    Returns:
        The calculated r1 regularization which can be used as the gradient penalty.
    Usage:
        inputs.requires_grad_(True)                 # Must turn on requires_grad for inputs.
        logits = discriminator(inputs)              # Pass through the discriminator.
        r1_reg = r1_regularization(inputs, logits)  # Compute r1 regularization.
    """
    r1 = torch.autograd.grad(
        outputs=[logits.sum()],
        inputs=[inputs],
        create_graph=True,
        only_inputs=True,
    )[0]
    r1 = r1.square()
    r1 = r1.sum(list(range(1, r1.ndim)))
    r1 = r1.mean()
    return 0.5 * r1
