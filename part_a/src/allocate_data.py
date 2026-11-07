"""Part A data allocation (version sydney-ml-v2). Copy this file UNCHANGED into src/allocate_data.py.

It selects exactly 3,000 real records from the supplied master CSV, using your numeric student ID
as the seed. It uses only the Python standard library.
"""
import csv                                   # read and write CSV files
import hashlib                               # SHA-256 fingerprints
import io                                    # treat text in memory like a file
import json                                  # write the manifest; build a stable row key
import re                                    # check that the student ID has digits only
from collections import defaultdict          # a dictionary of lists, one list per rating band
from pathlib import Path                     # file paths that work on Windows, macOS and Linux


def rating_band(value):
    """Return the rating band of one rating_number_historical value (text), or __MISSING__."""
    text = value.strip()                     # remove spaces around the value
    if not text:                             # an empty cell means the rating is missing
        return "__MISSING__"
    rating = float(text)                     # ratings are written like 3.7
    if rating < 2.5:                         # the five bands use the 2.5 / 3.5 / 4.0 / 4.5 cut points
        return "1_below_2.5"
    if rating < 3.5:
        return "2_2.5_to_3.4"
    if rating < 4.0:
        return "3_3.5_to_3.9"
    if rating < 4.5:
        return "4_4.0_to_4.4"
    return "5_4.5_and_above"


def allocate_student_data(student_id, csv_path="data/raw/zomato_rating_ml.csv",
                          output_dir="data/allocated"):
    """Select exactly 3,000 real records using a numeric student-ID seed."""
    version = "sydney-ml-v2"                 # the allocation version for this master file
    n = 3000                                 # every student gets 3,000 records
    expected_source_sha256 = (               # the fingerprint of the supplied master CSV
        "39350cb09da227bd289581347f3f20466eb326b5dc88b547a263a8e69b126289"
    )
    sid = str(student_id)                    # accept the ID as text or as a number
    if not re.fullmatch(r"[0-9]+", sid):     # digits only: no "s" prefix, spaces or signs
        raise ValueError("Student ID must contain digits 0-9 only.")
    sid = str(int(sid))                      # "0012345" and "12345" become the same ID
    if sid == "0":
        raise ValueError("Use your actual non-zero student ID.")
    seed = int(sid)                          # the ID is the seed
    source_bytes = Path(csv_path).read_bytes()               # read the master file exactly as stored
    source_hash = hashlib.sha256(source_bytes).hexdigest()   # its fingerprint
    if source_hash != expected_source_sha256:                # stop if the file was changed or resaved
        raise ValueError(
            "Master CSV differs from the assessment version. Download the "
            "original supplied file; do not edit or resave it before allocation."
        )
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")))   # one dict per row
    fields = reader.fieldnames                                               # the column names
    if not fields or "rating_number_historical" not in fields or "record_id" not in fields:
        raise ValueError("Master CSV is missing required columns.")
    groups = defaultdict(list)               # rating band -> list of (row key, row)
    seen = set()                             # rows already seen, to skip exact duplicates
    input_rows = 0                           # how many rows the master has
    for row in reader:
        input_rows += 1
        values = tuple(row[c] for c in fields)               # the whole row as a tuple
        if values in seen:                                   # skip an exact duplicate row
            continue
        seen.add(values)
        # Stable content identity; no source columns are added or changed.
        row_key = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        label = rating_band(row["rating_number_historical"])  # the stratum of this row
        groups[label].append((row_key, row))
    total = len(seen)                        # number of distinct rows
    if total < n:
        raise ValueError("Fewer than 3,000 distinct source rows are available.")
    labels = sorted(groups)                  # the bands in a fixed order
    # Largest-remainder proportional allocation, using integer arithmetic.
    quotas = {label: n * len(groups[label]) // total for label in labels}
    remaining = n - sum(quotas.values())     # rows still to hand out after rounding down
    priority = sorted(labels, key=lambda label: (
        -(n * len(groups[label]) % total), label           # largest remainder first, then by name
    ))
    for label in priority[:remaining]:
        quotas[label] += 1

    def rank(row_key, purpose):
        """A seeded, repeatable pseudo-random rank for one row."""
        message = f"{version}|{seed}|{purpose}|{row_key}"
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    selected = []                            # the chosen rows
    for label in labels:                     # within each band, take the rows with the lowest rank
        ranked = sorted(groups[label], key=lambda item: (
            rank(item[0], "select"), item[0]
        ))
        selected.extend(ranked[:quotas[label]])
    selected.sort(key=lambda item: (rank(item[0], "order"), item[0]))   # shuffle the final order
    assert len(selected) == n
    buffer = io.StringIO(newline="")         # write the allocation in memory first
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(row for _, row in selected)
    allocated_bytes = buffer.getvalue().encode("utf-8")
    dataset_hash = hashlib.sha256(allocated_bytes).hexdigest()           # fingerprint of your 3,000 rows
    code_input = f"{version}|{sid}|{source_hash}|{dataset_hash}"
    code = "SYD-" + hashlib.sha256(code_input.encode()).hexdigest()[:24].upper()   # verification code
    manifest = {
        "allocation_version": version,
        "student_id": sid,
        "allocation_seed": seed,
        "model_seed": seed % (2**32),        # a valid random_state for scikit-learn
        "source_sha256": source_hash,
        "dataset_sha256": dataset_hash,
        "source_rows": input_rows,
        "unique_source_rows": total,
        "exact_duplicates_removed": input_rows - total,
        "allocated_rows": n,
        "rating_group_counts": quotas,
        "verification_code": code,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)   # create data/allocated if needed
    dataset_path = out / f"student_restaurants_{sid}.csv"   # your 3,000 rows, named with your ID
    dataset_path.write_bytes(allocated_bytes)
    (out / "allocation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out / "verification_code.txt").write_text(code + "\n", encoding="utf-8")
    print(f"Allocated {n} rows: {dataset_path}")
    print(f"Dataset verification code: {code}")
    return dataset_path, manifest
