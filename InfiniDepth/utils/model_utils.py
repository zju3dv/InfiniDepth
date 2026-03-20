from InfiniDepth.model import MODEL_REGISTRY


def build_model(model_type: str, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}, options: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](**kwargs)

