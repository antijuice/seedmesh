"""Download the offline IP-to-ASN table used for network-diversity clustering.

Seedmesh prices anti-sybil influence in *network clusters* rather than peer count, because
peer ids are free and network diversity is not. Turning an IP into an autonomous system
number is what makes that real, and it has to happen **offline**: a DNS or whois lookup on
the routing hot path would add latency to every routing decision and hand an attacker a
denial-of-service lever.

Source: https://iptoasn.com — public data, no account or API key, v4 and v6 in one plain
sorted TSV. Rebuilt daily from BGP; re-fetch periodically, since ASN assignments change.

    python tools/fetch_asn_table.py --out data/ip2asn-combined.tsv.gz
    python tools/fetch_asn_table.py --out data/ip2asn-combined.tsv.gz --verify
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URL = "https://iptoasn.com/data/ip2asn-combined.tsv.gz"

# Well-known allocations, used to check the table is real and parsed correctly rather than
# merely present. If these fail, something is wrong with the download or the parser.
KNOWN = [
    ("8.8.8.8", 15169, "Google"),
    ("1.1.1.1", 13335, "Cloudflare"),
    ("9.9.9.9", 19281, "Quad9"),
    ("2606:4700:4700::1111", 13335, "Cloudflare v6"),
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "seedmesh-asn-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())
    print(f"wrote {destination} ({destination.stat().st_size / 2**20:.1f} MiB)")


def verify(path: Path) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from seedmesh.reputation.diversity import TableAsnResolver

    print(f"\nloading {path} ...")
    resolver = TableAsnResolver.from_file(path)
    print(f"loaded {len(resolver):,} routed ranges")

    failures = 0
    for address, expected_asn, label in KNOWN:
        actual = resolver.resolve(address)
        ok = actual == expected_asn
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {address:24s} -> AS{actual}"
              f"  (expected AS{expected_asn}, {label})")

    # A private address has no AS and must resolve to None, not to a spurious cluster.
    for address in ("192.168.1.1", "127.0.0.1", "10.0.0.1"):
        actual = resolver.resolve(address)
        ok = actual is None
        failures += not ok
        print(f"  {'OK  ' if ok else 'FAIL'}  {address:24s} -> {actual}  (expected None)")

    print(f"\n{len(KNOWN) + 3 - failures}/{len(KNOWN) + 3} checks passed")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ip2asn-combined.tsv.gz"))
    parser.add_argument("--url", default=URL)
    parser.add_argument("--verify", action="store_true", help="check known allocations after loading")
    parser.add_argument("--skip-download", action="store_true", help="verify an existing file")
    args = parser.parse_args()

    if not args.skip_download:
        try:
            download(args.url, args.out)
        except Exception as exc:
            print(f"download failed: {type(exc).__name__}: {exc}")
            return 2

    if args.verify:
        return verify(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
