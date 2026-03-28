class ZhivexAIError(Exception):
    pass


class ConfigurationError(ZhivexAIError):
    pass


class ValidationError(ZhivexAIError):
    pass


class UnsupportedFeatureError(ZhivexAIError):
    pass


class ParseError(ZhivexAIError):
    pass


class ProviderHTTPError(ZhivexAIError):
    def __init__(self, message: str, status: int, *, response_body: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.response_body = response_body
