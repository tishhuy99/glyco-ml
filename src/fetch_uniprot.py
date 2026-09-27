"""
Fetch human glycoproteins from UniProt with CARBOHYD annotations.

Writes data/raw/uniprot_glyco.json  — raw API records (one JSON array)
Writes data/raw/fetch_meta.json     — provenance for the methods section

UniProt REST API, no key needed. Uses the stream endpoint with pagination.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

API = "https://rest.uniprot.org/uniprotkb/search"

# Reviewed (SwissProt) human entries that carry at least one CARBOHYD feature.
QUERY = "(organism_id:9606) AND (reviewed:true) AND (ft_carbohyd:*)"

FIELDS = ",".join([
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "sequence",
    "ft_carbohyd",
    "cc_subcellular_location",
    "length",
])

PAGE_SIZE = 500


def fetch_all(query: str = QUERY) -> list[dict]:
    records: list[dict] = []
    params = {
        "query": query,
        "fields": FIELDS,
        "format": "json",
        "size": PAGE_SIZE,
    }
    url = API
    pbar = tqdm(desc="fetching", unit="rec")
    while url:
        resp = requests.get(url, params=params if url == API else None, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("results", [])
        records.extend(batch)
        pbar.update(len(batch))

        # UniProt paginates via a Link header with rel="next"
        url = resp.links.get("next", {}).get("url")
        params = None
        if url:
            time.sleep(0.2)  # be polite
    pbar.close()
    return records


def main() -> None:
    records = fetch_all()

    out = RAW / "uniprot_glyco.json"
    out.write_text(json.dumps(records))

    meta = {
        "source": "UniProtKB REST API",
        "endpoint": API,
        "query": QUERY,
        "fields": FIELDS,
        "n_records": len(records),
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
    }
    (RAW / "fetch_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {len(records)} records -> {out}")
    print(f"provenance -> {RAW / 'fetch_meta.json'}")


if __name__ == "__main__":
    main()
