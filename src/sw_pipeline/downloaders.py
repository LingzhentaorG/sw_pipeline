from __future__ import annotations

import ftplib
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


DEFAULT_USER_AGENT = "sw-pipeline/0.1"


@dataclass(frozen=True)
class DownloadResult:
    path: Path | None
    status: str
    error: str | None
    attempts: int
    protocol: str


def create_retry_session(total: int = 4, backoff: float = 0.8, user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total,
        connect=total,
        read=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def infer_protocol(url: str, transport: str | None = None) -> str:
    requested = str(transport or "auto").lower()
    if requested in {"http", "https", "ftp"}:
        return requested
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in {"http", "https", "ftp"}:
        return scheme
    return "https"


def create_earthdata_session(
    username: str,
    password: str,
    *,
    total: int = 4,
    backoff: float = 0.8,
    verify: bool = True,
) -> requests.Session:
    session = create_retry_session(total=total, backoff=backoff)
    if not username or not password:
        return session
    session.auth = (username, password)
    session.get("https://urs.earthdata.nasa.gov", timeout=60, verify=verify, allow_redirects=True).raise_for_status()
    return session


def fetch_text(
    url: str,
    *,
    session: requests.Session | None = None,
    timeout: int = 60,
    verify: bool = True,
) -> str:
    own_session = session is None
    session = session or create_retry_session()
    try:
        response = session.get(url, timeout=timeout, verify=verify, allow_redirects=True)
        response.raise_for_status()
        return response.text
    finally:
        if own_session:
            session.close()


def download_to_path(
    url: str,
    target: Path,
    *,
    transport: str = "auto",
    session: requests.Session | None = None,
    timeout: int = 90,
    verify: bool = True,
    max_retries: int = 4,
    temp_suffix: str = ".part",
    auth: tuple[str, str] | None = None,
) -> DownloadResult:
    protocol = infer_protocol(url, transport)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return DownloadResult(path=target, status="ok", error=None, attempts=0, protocol=protocol)

    last_error: str | None = None
    attempts = 0
    for attempt in range(1, max_retries + 1):
        attempts = attempt
        try:
            if protocol == "ftp":
                _download_ftp(url, target, timeout=timeout, temp_suffix=temp_suffix, auth=auth)
            else:
                _download_http(
                    url,
                    target,
                    session=session,
                    timeout=timeout,
                    verify=verify,
                    temp_suffix=temp_suffix,
                )
            return DownloadResult(path=target, status="ok", error=None, attempts=attempts, protocol=protocol)
        except requests.HTTPError as exc:
            last_error = str(exc)
            _cleanup_partial(target, temp_suffix)
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                break
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 16))
        except (ftplib.Error, EOFError, OSError, requests.RequestException) as exc:
            last_error = str(exc)
            _cleanup_partial(target, temp_suffix)
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 16))
    return DownloadResult(path=None, status="error", error=last_error, attempts=attempts, protocol=protocol)


def _download_http(
    url: str,
    target: Path,
    *,
    session: requests.Session | None,
    timeout: int,
    verify: bool,
    temp_suffix: str,
) -> None:
    own_session = session is None
    session = session or create_retry_session()
    temp_path = _temp_path(target, temp_suffix)
    try:
        with session.get(url, stream=True, timeout=timeout, verify=verify, allow_redirects=True) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise OSError("Downloaded file is empty")
        temp_path.replace(target)
    finally:
        if own_session:
            session.close()


def _download_ftp(
    url: str,
    target: Path,
    *,
    timeout: int,
    temp_suffix: str,
    auth: tuple[str, str] | None,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "ftp":
        raise ValueError(f"Invalid FTP URL: {url}")
    temp_path = _temp_path(target, temp_suffix)
    username = auth[0] if auth else "anonymous"
    password = auth[1] if auth else "anonymous@"
    remote_path = parsed.path
    with ftplib.FTP() as ftp:
        ftp.connect(parsed.hostname, parsed.port or 21, timeout=timeout)
        ftp.login(user=username, passwd=password)
        with temp_path.open("wb") as handle:
            ftp.retrbinary(f"RETR {remote_path}", handle.write)
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise OSError("Downloaded file is empty")
    temp_path.replace(target)


def _temp_path(target: Path, temp_suffix: str) -> Path:
    try:
        target.unlink(missing_ok=True)
    except PermissionError:
        pass
    return target.with_name(f"{target.name}{temp_suffix}.{time.time_ns()}")


def _cleanup_partial(target: Path, temp_suffix: str) -> None:
    try:
        target.unlink(missing_ok=True)
    except PermissionError:
        pass
    for path in target.parent.glob(f"{target.name}{temp_suffix}*"):
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            continue
