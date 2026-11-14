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
