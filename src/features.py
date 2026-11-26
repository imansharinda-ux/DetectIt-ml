import math
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_WORDS = (
    "login", "signin", "verify", "account", "update", "secure",
    "bank", "confirm", "password", "wallet", "free", "bonus",
)

IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

SUSPICIOUS_TLDS = {
    "top", "xyz", "tk", "ml", "ga", "cf", "gq", "ru", "cn", "info",
    "biz", "club", "online", "site", "live", "app", "dev", "io", "co",
}

COMMON_TLDS = {"com", "org", "net", "edu", "gov", "uk", "de", "jp", "fr", "au", "ca", "it", "nl"}

SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly"}

BRANDS = (
    "paypal", "apple", "google", "microsoft", "amazon", "facebook", "netflix",
    "bank", "ebay", "outlook", "office", "instagram", "whatsapp", "dropbox",
)

FILE_EXTENSION_PATTERN = re.compile(r"\.(php|html?|asp|aspx|jsp|exe|zip|cgi)$")

SCHEME_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)

WWW_PATTERN = re.compile(r"^www\.", re.IGNORECASE)

SCHEME_FEATURES = ["HasHTTPSScheme", "HasWWW"]


def normalize_url(url):
    return str(url).strip().lower().rstrip("/")


def clean_url(url):
    return WWW_PATTERN.sub("", SCHEME_PATTERN.sub("", str(url).strip()))


def shannon_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def extract_url_features(url):
    url = str(url).strip().lower()
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or ""
    path = parsed.path or ""
    host_is_ip = bool(IP_PATTERN.match(host))

    return {
        "FullURLLength": len(url),
        "HostLength": len(host),
        "PathLength": len(path),
        "QueryLength": len(parsed.query),
        "PathDepth": len([part for part in path.split("/") if part]),
        "NoOfDotsInURL": url.count("."),
        "NoOfHyphensInURL": url.count("-"),
        "NoOfSlashesInURL": url.count("/"),
        "NoOfDigitsInHost": sum(ch.isdigit() for ch in host),
        "HostSubdomainCount": 0 if host_is_ip else max(host.count(".") - 1, 0),
        "SuspiciousWordCount": sum(url.count(word) for word in SUSPICIOUS_WORDS),
        "URLEntropy": shannon_entropy(url),
        "HostEntropy": shannon_entropy(host),
        "HasAtSymbol": int("@" in url),
        "HasHTTPSScheme": int(parsed.scheme == "https"),
        "HostIsIP": int(host_is_ip),
        "HostHasHyphen": int("-" in host),
        "HasDoubleSlashInPath": int("//" in path),
    }


def build_url_features(urls):
    return pd.DataFrame([extract_url_features(url) for url in urls], index=urls.index)


def has_port(parsed):
    try:
        return int(parsed.port is not None)
    except ValueError:
        return 0


def extract_extra_url_features(url):
    raw_url = str(url).strip()
    url = raw_url.lower()
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or ""
    path = parsed.path or ""

    host_parts = [part for part in host.split(".") if part]
    tld = host_parts[-1] if host_parts else ""
    registered_domain = ".".join(host_parts[-2:]) if len(host_parts) >= 2 else host
    host_tokens = [token for token in re.split(r"[.\-]", host) if token]
    path_tokens = [token for token in re.split(r"[/\-_.?=&]", path) if token]

    url_length = max(len(url), 1)
    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    special_chars = sum(not ch.isalnum() for ch in url)
    host_letters = max(sum(ch.isalpha() for ch in host), 1)

    return {
        "TLDLength": len(tld),
        "IsSuspiciousTLD": int(tld in SUSPICIOUS_TLDS),
        "IsCommonTLD": int(tld in COMMON_TLDS),
        "NoOfDigitsInURL": digits,
        "DigitRatioInURL": digits / url_length,
        "LetterRatioInURL": letters / url_length,
        "SpecialCharRatioInURL": special_chars / url_length,
        "NoOfUppercaseInURL": sum(ch.isupper() for ch in raw_url),
        "NoOfQueryParams": len([param for param in parsed.query.split("&") if param]),
        "NoOfPercentInURL": url.count("%"),
        "NoOfUnderscoreInURL": url.count("_"),
        "HasPort": has_port(parsed),
        "HasWWW": int(host.startswith("www.")),
        "HostTokenCount": len(host_tokens),
        "LongestHostToken": max((len(token) for token in host_tokens), default=0),
        "LongestPathToken": max((len(token) for token in path_tokens), default=0),
        "HasFileExtension": int(bool(FILE_EXTENSION_PATTERN.search(path))),
        "IsShortener": int(host in SHORTENERS),
        "BrandOutsideDomain": int(any(brand in url and brand not in registered_domain for brand in BRANDS)),
        "HasPunycode": int("xn--" in host),
        "HostVowelRatio": sum(ch in "aeiou" for ch in host) / host_letters,
        "MaxConsecutiveDigits": max((len(match) for match in re.findall(r"\d+", url)), default=0),
    }


def build_all_url_features(urls):
    extra = pd.DataFrame([extract_extra_url_features(url) for url in urls], index=urls.index)
    return pd.concat([build_url_features(urls), extra], axis=1)


def build_clean_url_features(urls):
    return build_all_url_features(urls.map(clean_url)).drop(columns=SCHEME_FEATURES)


def build_page_features(df):
    total_refs = df["NoOfSelfRef"] + df["NoOfExternalRef"] + df["NoOfEmptyRef"]
    title = df["Title"].fillna("").astype(str)

    return pd.DataFrame({
        "ExternalRefRatio": df["NoOfExternalRef"] / (total_refs + 1),
        "EmptyRefRatio": df["NoOfEmptyRef"] / (total_refs + 1),
        "TotalResources": df["NoOfImage"] + df["NoOfCSS"] + df["NoOfJS"],
        "PageQualityScore": df[["HasTitle", "HasFavicon", "HasDescription", "HasCopyrightInfo", "HasSocialNet", "IsResponsive", "Robots"]].sum(axis=1),
        "HasSensitiveForm": df[["HasPasswordField", "HasExternalFormSubmit", "HasHiddenFields"]].max(axis=1),
        "HasFinancialKeyword": df[["Bank", "Pay", "Crypto"]].max(axis=1),
        "TitleLength": title.str.len(),
        "TitleWordCount": title.str.split().str.len().fillna(0).astype(int),
    }, index=df.index)
