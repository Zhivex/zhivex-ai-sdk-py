"""Technical detectors for Latin American identifiers and payment cards.

A match is removed only when its check digit is valid. This is not legal advice
and not a determination of compliance with Argentina's Personal Data Protection
Act or Brazil's LGPD.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .safety import RedactionPolicy, RedactionRule, create_redaction_policy

LatamCountry = str
_COUNTRIES = ("AR", "BR", "CL", "MX")
_CUIT_PREFIXES = {"20", "23", "24", "27", "30", "33", "34"}
_CUIT_WEIGHTS = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
_CBU_WEIGHTS_A = (7, 1, 3, 9, 7, 1, 3)
_CBU_WEIGHTS_B = (3, 9, 7, 1, 3, 9, 7, 1, 3, 9, 7, 1, 3)
_CNPJ_WEIGHTS_A = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_CNPJ_WEIGHTS_B = (6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_RFC_ALPHABET = "0123456789ABCDEFGHIJKLMN&OPQRSTUVWXYZ Ñ"
_CURP_ALPHABET = "0123456789ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
_CURP_PATTERN = re.compile(
    r"^[A-Z][AEIOUX][A-Z]{2}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[HM]"
    r"(?:AS|BC|BS|CC|CL|CM|CS|CH|DF|DG|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)"
    r"[B-DF-HJ-NP-TV-Z]{3}[A-Z0-9]\d$"
)


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _weighted(digits: str, weights: Sequence[int]) -> int:
    return sum(int(digits[index]) * weight for index, weight in enumerate(weights))


def _cuit_check_digit(first10: str) -> int:
    check = 11 - (_weighted(first10, _CUIT_WEIGHTS) % 11)
    if check == 11:
        return 0
    if check == 10:
        return 9
    return check


def _is_valid_cuit(value: str) -> bool:
    if re.fullmatch(r"\d{2}-\d{8}-\d", value) is None:
        return False
    digits = _digits(value)
    if digits[:2] not in _CUIT_PREFIXES:
        return False
    return _cuit_check_digit(digits[:10]) == int(digits[10])


def _mod10(digits: str, weights: Sequence[int]) -> int:
    return (10 - (_weighted(digits, weights) % 10)) % 10


def _is_valid_cbu(value: str) -> bool:
    if re.fullmatch(r"\d{22}", value) is None or value.startswith("000"):
        return False
    return _mod10(value[:7], _CBU_WEIGHTS_A) == int(value[7]) and _mod10(value[8:21], _CBU_WEIGHTS_B) == int(value[21])


def _is_valid_cvu(value: str) -> bool:
    if re.fullmatch(r"000\d{19}", value) is None:
        return False
    return _mod10(value[:7], _CBU_WEIGHTS_A) == int(value[7]) and _mod10(value[8:21], _CBU_WEIGHTS_B) == int(value[21])


def _repeated(digits: str) -> bool:
    return re.fullmatch(r"(\d)\1+", digits) is not None


def _trailing_check(base: str) -> int:
    total = sum(int(base[index]) * (len(base) + 1 - index) for index in range(len(base)))
    remainder = total % 11
    return 0 if remainder < 2 else 11 - remainder


def _cpf_check_digits(first9: str) -> str:
    first = _trailing_check(first9)
    second = _trailing_check(f"{first9}{first}")
    return f"{first}{second}"


def _is_valid_cpf(value: str) -> bool:
    if re.fullmatch(r"\d{3}\.\d{3}\.\d{3}-\d{2}", value) is None:
        return False
    digits = _digits(value)
    if _repeated(digits):
        return False
    return _cpf_check_digits(digits[:9]) == digits[9:]


def _cnpj_digit(base: str) -> int:
    weights = _CNPJ_WEIGHTS_A if len(base) == 12 else _CNPJ_WEIGHTS_B
    remainder = _weighted(base, weights) % 11
    return 0 if remainder < 2 else 11 - remainder


def _cnpj_check_digits(first12: str) -> str:
    first = _cnpj_digit(first12)
    second = _cnpj_digit(f"{first12}{first}")
    return f"{first}{second}"


def _is_valid_cnpj(value: str) -> bool:
    if re.fullmatch(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", value) is None:
        return False
    digits = _digits(value)
    if _repeated(digits):
        return False
    return _cnpj_check_digits(digits[:12]) == digits[12:]


def _rut_check_digit(body: str) -> str:
    total = 0
    factor = 2
    for index in range(len(body) - 1, -1, -1):
        total += int(body[index]) * factor
        factor = 2 if factor == 7 else factor + 1
    check = 11 - (total % 11)
    if check == 11:
        return "0"
    if check == 10:
        return "K"
    return str(check)


def _is_valid_rut(value: str) -> bool:
    match = re.fullmatch(r"(\d{1,2})\.(\d{3})\.(\d{3})-([\dkK])", value)
    if match is None:
        return False
    body = f"{match.group(1)}{match.group(2)}{match.group(3)}"
    return _rut_check_digit(body) == match.group(4).upper()


def _curp_check_digit(body17: str) -> str:
    total = 0
    for index, character in enumerate(body17):
        value = _CURP_ALPHABET.find(character)
        if value < 0:
            return ""
        total += value * (18 - index)
    return str((10 - (total % 10)) % 10)


def _is_valid_curp(value: str) -> bool:
    if _CURP_PATTERN.fullmatch(value) is None:
        return False
    return _curp_check_digit(value[:17]) == value[17]


def _rfc_check_digit(body: str) -> str:
    weighted = f" {body}" if len(body) == 11 else body
    total = 0
    for index, character in enumerate(weighted):
        value = _RFC_ALPHABET.find(character)
        if value < 0:
            return ""
        total += value * (len(weighted) + 1 - index)
    check = 11 - (total % 11)
    if check == 11:
        return "0"
    if check == 10:
        return "A"
    return str(check)


def _is_valid_rfc(value: str) -> bool:
    match = re.fullmatch(r"([A-ZÑ&]{3,4})(\d{2})(\d{2})(\d{2})([A-Z0-9]{3})", value)
    if match is None:
        return False
    month = int(match.group(3))
    day = int(match.group(4))
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return False
    return _rfc_check_digit(value[:-1]) == value[-1]


def _luhn(digits: str) -> bool:
    if re.fullmatch(r"\d{13,19}", digits) is None or _repeated(digits):
        return False
    total = 0
    alternate = False
    for character in reversed(digits):
        current = int(character)
        if alternate:
            current *= 2
            if current > 9:
                current -= 9
        total += current
        alternate = not alternate
    return total % 10 == 0


def _is_valid_card(value: str) -> bool:
    separated = bool(re.search(r"[ -]", value))
    digits = _digits(value)
    if not separated and len(digits) != 16:
        return False
    if len(digits) < 13 or len(digits) > 19:
        return False
    return _luhn(digits)


def _keep(match: str, valid: bool, label: str) -> str:
    return f"[REDACTED:{label}]" if valid else match


def _rule(name: str, pattern: str, validate: Callable[[str], bool], label: str) -> RedactionRule:
    def replacement(match: str) -> str:
        return _keep(match, validate(match), label)

    return RedactionRule(pattern, replacement, name)


def _country_rules(countries: set[str]) -> list[RedactionRule]:
    rules: list[RedactionRule] = []
    if "AR" in countries:
        rules.append(_rule("cuit", r"(?<!\d)\d{2}-\d{8}-\d(?!\d)", _is_valid_cuit, "CUIT"))

        def cbu_replacement(match: str) -> str:
            if _is_valid_cvu(match):
                return "[REDACTED:CVU]"
            if _is_valid_cbu(match):
                return "[REDACTED:CBU]"
            return match

        rules.append(RedactionRule(r"(?<!\d)\d{22}(?!\d)", cbu_replacement, "cbu"))
    if "BR" in countries:
        rules.append(_rule("cnpj", r"(?<!\d)\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}(?!\d)", _is_valid_cnpj, "CNPJ"))
        rules.append(_rule("cpf", r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)", _is_valid_cpf, "CPF"))
    if "CL" in countries:
        rules.append(_rule("rut", r"(?<!\d)\d{1,2}\.\d{3}\.\d{3}-[\dkK](?!\d)", _is_valid_rut, "RUT"))
    if "MX" in countries:
        rules.append(_rule(
            "curp",
            r"(?<![A-Z0-9])[A-Z][AEIOUX][A-Z]{2}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[HM][A-Z]{2}[B-DF-HJ-NP-TV-Z]{3}[A-Z0-9]\d(?![A-Z0-9])",
            _is_valid_curp,
            "CURP",
        ))
        rules.append(_rule("rfc", r"(?<![A-Z0-9Ñ&])[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}(?![A-Z0-9Ñ&])", _is_valid_rfc, "RFC"))
    return rules


def create_latam_sensitive_data_policy(
    *,
    countries: Sequence[LatamCountry] | None = None,
    include_cards: bool = True,
    include_emails: bool = False,
    replacement: str = "[REDACTED]",
    rules: list[RedactionRule] | None = None,
) -> RedactionPolicy:
    selected = set(countries if countries is not None else _COUNTRIES)
    for country in selected:
        if country not in _COUNTRIES:
            raise TypeError(f'Unknown LatAm detector country "{country}".')
    compiled = [
        *_country_rules(selected),
        *([_rule("card", r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", _is_valid_card, "CARD")] if include_cards else []),
        *(rules or []),
    ]
    return create_redaction_policy(include_emails=include_emails, replacement=replacement, rules=compiled)
