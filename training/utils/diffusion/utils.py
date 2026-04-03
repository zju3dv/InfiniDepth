import torch


def randlike_shape(shape, generator):
    batch_size = shape[0]
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        x = torch.cat(
            [torch.randn(shape, generator=generator[i], device=generator[i].device) for i in range(batch_size)]
        )
    else:
        x = torch.randn(shape, generator=generator, device=generator.device)
    return x
