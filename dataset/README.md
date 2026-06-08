# Clean PK Nesting Dataset

This folder keeps only the data needed for the CZ402-on-PK multi-hide nesting experiment.

## Pieces
- `pieces/CZ402_pattern_cutting.xml`: CZ402 Pattern Cutting XML used for the 21 unique / 28 per-bag leather pieces.
- `pieces/piece_catalog_area_rule.csv`: piece catalog sorted by area descending. Pieces are sorted by area and split into four groups: largest 6 Q1, next 5 Q2, next 5 Q3, smallest 5 Q4.
- `pieces/piece_shapes_preview_area_rule.png`: visual preview of the 21 piece outlines, colored by provisional required quality.

## Leathers
- `leathers/001_...` through `leathers/030_...`: 30 hides sorted by ascending Q1 ratio.
- Each folder contains `leather_svg.xml` for parsing, optional stat/preview files, and `source_info.json`.

## Experiment order
`leather_manifest.csv` defines the exact hide usage order. Hide 001 has the lowest Q1 ratio and is used first.
