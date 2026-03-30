class MedFlowError(Exception):
    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(MedFlowError):
    pass


class ProviderError(MedFlowError):
    def __init__(self, message: str, *, provider: str | None = None, details: dict | None = None) -> None:
        d = dict(details or {})
        if provider:
            d["provider"] = provider
        super().__init__(message, details=d)


class OCRError(MedFlowError):
    def __init__(self, message: str, *, engine: str | None = None, details: dict | None = None) -> None:
        d = dict(details or {})
        if engine:
            d["engine"] = engine
        super().__init__(message, details=d)


class DeidentifyError(MedFlowError):
    pass


class IngestionError(MedFlowError):
    pass


class RetrievalError(MedFlowError):
    pass


class EvaluationError(MedFlowError):
    pass
