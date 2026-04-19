class ModelPromotionError(RuntimeError):
    """Raised when a trained model does not meet promotion criteria."""


class ModelNotReadyError(RuntimeError):
    """Raised when prediction is requested before a model is available."""