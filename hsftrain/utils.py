from pathlib import Path

import wget
import xxhash


def get_hash(fname: str) -> str:
    """
    Get xxHash3 of a file

    Args:
        fname (str): Path to file

    Returns:
        str: xxHash3 of file
    """
    xxh = xxhash.xxh3_64()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            xxh.update(chunk)
    return xxh.hexdigest()


def fetch(directory: str, filename: str, url: str, xxh3_64: str) -> None:
    """
    Fetch a model from a url

    Args:
        directory (str): Directory to save model
        filename (str): Filename of model
        url (str): Url to download model from
        xxh3_64 (str): xxh3_64 of model
    """
    p = Path(directory).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    outfile = p / filename

    if outfile.exists():
        if get_hash(str(outfile)) == xxh3_64:
            print(f"{filename} already exists and is up to date")
            return
        else:
            print(f"{filename} already exists but is not up to date")
            outfile.unlink()

    print(f"Fetching {url}")
    wget.download(url, out=str(outfile))
    print("\n")

    hash = get_hash(str(outfile))
    if not xxh3_64 == hash:
        outfile.unlink()
        raise Exception(
            f"xxh3_64 checksum failed, expected {xxh3_64}, got {hash}")
