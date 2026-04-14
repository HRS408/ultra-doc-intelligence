import fitz
import re

def parse_pdf_old(path: str) -> str:
    """Original method — column-by-column"""
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def parse_pdf_new(path: str) -> str:
    doc = fitz.open(path)
    pages = []

    for page in doc:
        # span-level extraction — much more granular than blocks
        data = page.get_text("dict")
        
        # collect all spans with their x,y positions
        spans = []
        for block in data["blocks"]:
            if block.get("type") != 0:  # skip image blocks
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    x0 = round(span["bbox"][0])
                    y0 = round(span["bbox"][1])
                    spans.append((y0, x0, text))

        # sort top-to-bottom, left-to-right
        spans.sort(key=lambda s: (s[0], s[1]))

        # group spans into rows — same row = y within 6px of each other
        rows = []
        current_row = []
        prev_y = None

        for y0, x0, text in spans:
            if prev_y is None or abs(y0 - prev_y) <= 6:
                current_row.append((x0, text))
                prev_y = y0 if prev_y is None else (prev_y + y0) / 2
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [(x0, text)]
                prev_y = y0

        if current_row:
            rows.append(current_row)

        # detect column layout
        # if a row has 2+ spans, check if they're in distinct x-zones
        # left zone: x < page_width/2, right zone: x >= page_width/2
        page_width = page.rect.width
        lines_out = []

        for row in rows:
            if len(row) == 1:
                lines_out.append(row[0][1])
            else:
                # sort by x within row
                row.sort(key=lambda r: r[0])
                # join with separator
                joined = "  |  ".join(text for _, text in row)
                lines_out.append(joined)

        pages.append("\n".join(lines_out))

    return "\n\n".join(pages)

# ── Run on all three docs ─────────────────────────────────────────────
docs = {
    "carrier_rc":  "docs/LD53657-Carrier-RC.pdf",
    "shipper_rc":  "docs/LD53657-Shipper-RC.pdf",
    "bol":         "docs/BOL53657_billoflading.pdf",
}

for name, path in docs.items():
    old = parse_pdf_old(path)
    new = parse_pdf_new(path)

    # Save both versions for manual inspection
    with open(f"docs/{name}_old.txt", "w") as f:
        f.write(old)
    with open(f"docs/{name}_new.txt", "w") as f:
        f.write(new)

    # Quick targeted checks — does label+value now appear on same line?
    checks = {
        "carrier_rc": [
            ("Flatbed",              "Equipment+value on same line"),
            ("MC1685682",            "MC number present"),
            ("SWIFT SHIFT",          "Carrier name present"),
            ("400",                  "Rate present"),
            ("John Doe",             "Driver name present"),
            ("123456",               "Truck number present"),
        ],
        "shipper_rc": [
            ("Test ABC",             "Customer name present"),
            ("1000",                 "Agreed amount present"),
            ("Fontana",              "Consignee address present"),
            ("56000",                "Weight present"),
        ],
        "bol": [
            ("LD53657",              "Load ID present"),
            ("112233ABC",            "PO number present"),
            ("64000",                "COD value present"),
            ("10000",                "Unit count present"),
        ],
    }

    print(f"\n{'='*55}")
    print(f"  {name.upper()}")
    print(f"  Old: {len(old)} chars  →  New: {len(new)} chars")
    print(f"{'='*55}")

    all_passed = True
    for keyword, description in checks[name]:
        in_old = keyword.lower() in old.lower()
        in_new = keyword.lower() in new.lower()
        status = "✓" if in_new else "✗ MISSING"
        change = "(was missing in old too)" if not in_old and not in_new else \
                 "(FIXED)"                  if not in_old and in_new     else \
                 "(still present)"          if in_old and in_new         else \
                 "(REGRESSION — was in old!)"
        print(f"  {status}  {description:35s} {change}")
        if not in_new:
            all_passed = False

    print(f"\n  {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED — see above'}")

    # Show side-by-side preview of first 400 chars
    print(f"\n  --- OLD (first 400 chars) ---")
    print("  " + old[:400].replace("\n", "\n  "))
    print(f"\n  --- NEW (first 400 chars) ---")
    print("  " + new[:400].replace("\n", "\n  "))