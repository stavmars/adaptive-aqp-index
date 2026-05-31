#!/usr/bin/env bash
# Filter the US eBird Basic Dataset (EBD) down to the benchmark subset:
# continental US (CONUS) rows observed in 2023-2026. Excludes Alaska, Hawaii,
# and overseas territories via a lat/lon bounding box. The header row is
# preserved, so the eBird column names carry through to the Parquet step.
#
# Columns (1-indexed, tab-delimited, standard EBD layout):
#   29 = LATITUDE
#   30 = LONGITUDE
#   31 = OBSERVATION DATE (YYYY-MM-DD)
#
# CONUS bounding box (generous margin):
#   Latitude:  24.0 .. 50.0    (southern tip of the FL Keys to the Canadian border)
#   Longitude: -130.0 .. -60.0 (Pacific coast to Atlantic coast)
#
# Usage:
#   bash scripts/filter_ebird_us.sh <input_ebd.txt> <output.txt>
#
# Input  : the full US EBD release (.txt, tab-delimited, ~460 GB / ~1.25B rows).
# Output : the filtered benchmark TSV (~155 GB / ~420M rows). Run
#          tools/csv_to_parquet --has-header --delimiter tab --null-string X on
#          it to produce the Parquet referenced by configs/datasets/ebird_us.yaml.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_ebd.txt> <output.txt>" >&2
    exit 2
fi

INPUT="$1"
OUTPUT="$2"

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: Input file not found: $INPUT" >&2
    exit 1
fi

echo "Filtering US eBird: $INPUT"
echo "Output: $OUTPUT"
echo "Keeping: years 2023-2026, CONUS only (lat 24-50, lon -130 to -60)"
echo "Started: $(date)"

# Preserve the header line.
head -1 "$INPUT" > "$OUTPUT"

# Filter: observation year in 2023-2026 AND inside the CONUS bounding box.
# awk columns (1-indexed): $29=LATITUDE, $30=LONGITUDE, $31=OBSERVATION DATE.
tail -n +2 "$INPUT" | awk -F'\t' '
    $31 ~ /^202[3-6]/ && $29+0 >= 24 && $29+0 <= 50 && $30+0 >= -130 && $30+0 <= -60
' >> "$OUTPUT"

ROWS=$(wc -l < "$OUTPUT")
SIZE=$(du -h "$OUTPUT" | cut -f1)
echo "Done: $(date)"
echo "Output rows: $ROWS (including header)"
echo "Output size: $SIZE"
