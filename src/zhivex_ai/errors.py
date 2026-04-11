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

    def __str__(self) -> str:
        message = super().__str__()
        if not self.response_body:
            return message
        body = self.response_body.strip()
        if not body:
            return message
        compact = " ".join(body.split())
        snippet = compact if len(compact) <= 500 else f"{compact[:497]}..."
        return f"{message} Response body: {snippet}"
