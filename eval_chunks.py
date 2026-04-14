import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

# ── New parser (span-level) ──────────────────────────────────────────
def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        data = page.get_text("dict")
        spans = []
        for block in data["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        spans.append((round(span["bbox"][1]), round(span["bbox"][0]), text))
        spans.sort(key=lambda s: (s[0], s[1]))
        rows, current_row, prev_y = [], [], None
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
        lines_out = []
        for row in rows:
            row.sort(key=lambda r: r[0])
            lines_out.append("  |  ".join(t for _, t in row) if len(row) > 1 else row[0][1])
        pages.append("\n".join(lines_out))
    return "\n\n".join(pages)

# ── Improvement 1: prefix injection ─────────────────────────────────
def add_prefix(chunk: str, doc_type: str, load_id: str) -> str:
    return f"[{doc_type.upper()}] [Load: {load_id}]\n{chunk}"

# ── Improvement 2: small doc guard ──────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    if len(text) <= 1200:
        return [text]                      # BOL stays as one chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

# ── Load docs ────────────────────────────────────────────────────────
carrier_rc = parse_pdf("docs/LD53657-Carrier-RC.pdf")
shipper_rc = parse_pdf("docs/LD53657-Shipper-RC.pdf")
bol        = parse_pdf("docs/BOL53657_billoflading.pdf")

tests = {
    "carrier_rc": {
        "text":    carrier_rc,
        "doc_type": "carrier_rc",
        "load_id":  "LD53657",
        "pairs": [
            ("What is the carrier rate?",            "400"),
            ("Who is the carrier?",                  "SWIFT SHIFT LOGISTICS"),
            ("What is the carrier MC number?",       "MC1685682"),
            ("What is the driver name?",             "John Doe"),
            ("What is the truck number?",            "123456"),
            ("What is the equipment type?",          "Flatbed"),
            ("What is the PO number at pickup?",     "112233ABC"),
            ("What is the pickup appointment time?", "15:00"),
        ]
    },
    "shipper_rc": {
        "text":    shipper_rc,
        "doc_type": "shipper_rc",
        "load_id":  "LD53657",
        "pairs": [
            ("What is the agreed amount?",           "1000"),
            ("Who is the customer?",                 "Test ABC"),
            ("What is the shipper location?",        "Los Angeles"),
            ("What is the consignee address?",       "Fontana"),
            ("What is the commodity weight?",        "56000"),
        ]
    },
    "bol": {
        "text":    bol,
        "doc_type": "bol",
        "load_id":  "LD53657",
        "pairs": [
            ("What is the load ID?",                 "LD53657"),
            ("What is the COD value?",               "64000"),
            ("How many units?",                      "10000"),
            ("What is the PO number?",               "112233ABC"),
            ("Who is the shipper?",                  "AAA"),
            ("Who is the consignee?",                "xyz"),
        ]
    }
}

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def eval_config(text, pairs, doc_type, load_id, chunk_size, overlap, use_prefix):
    chunks = chunk_text(text, chunk_size, overlap)
    
    # Improvement 1: optionally inject prefix before embedding
    embed_chunks = [add_prefix(c, doc_type, load_id) for c in chunks] if use_prefix else chunks
    
    chunk_embeddings = model.encode(embed_chunks, show_progress_bar=False)
    
    hits, miss_details, sim_scores = 0, [], []
    for question, answer in pairs:
        q_emb = model.encode(question)
        scores = util.cos_sim(q_emb, chunk_embeddings)[0]
        top_idx = scores.argmax().item()
        top_score = float(scores[top_idx])
        top_chunk = chunks[top_idx]       # show original (not prefixed) for readability
        sim_scores.append(top_score)
        if answer.lower() in top_chunk.lower():
            hits += 1
        else:
            miss_details.append((question, answer, top_score, top_chunk[:100]))

    return {
        "chunk_size":   chunk_size,
        "overlap":      overlap,
        "prefix":       use_prefix,
        "num_chunks":   len(chunks),
        "hit_rate":     round(hits / len(pairs), 2),
        "avg_sim":      round(sum(sim_scores) / len(sim_scores), 3),   # ← confidence proxy
        "min_sim":      round(min(sim_scores), 3),                      # ← worst case
        "misses":       miss_details
    }

# ── Grid search: chunk size × overlap × prefix on/off ───────────────
print("\n" + "="*70)
print("FULL EVAL — hit rate + similarity scores (confidence proxy)")
print("="*70)

for doc_name, cfg in tests.items():
    print(f"\n── {doc_name.upper()} ──")
    results = []
    for size in [300, 400, 500, 600, 800]:
        for overlap in [0, 50, 100]:
            for prefix in [False, True]:
                r = eval_config(
                    cfg["text"], cfg["pairs"],
                    cfg["doc_type"], cfg["load_id"],
                    size, overlap, prefix
                )
                results.append(r)

    # sort by hit_rate first, then avg_sim
    top = sorted(results, key=lambda x: (-x["hit_rate"], -x["avg_sim"]))[:6]

    print(f"{'Size':>5} {'Ovlp':>5} {'Prefix':>7} {'Chunks':>7} "
          f"{'HitRate':>9} {'AvgSim':>8} {'MinSim':>8}")
    print("-" * 60)
    for r in top:
        prefix_flag = "YES" if r["prefix"] else "no"
        print(f"{r['chunk_size']:>5} {r['overlap']:>5} {prefix_flag:>7} "
              f"{r['num_chunks']:>7} {r['hit_rate']:>8.0%} "
              f"{r['avg_sim']:>8.3f} {r['min_sim']:>8.3f}")

    best = top[0]
    if best["misses"]:
        print(f"\n  Remaining misses (best config):")
        for q, exp, sim, chunk in best["misses"]:
            print(f"  ✗ '{q}'")
            print(f"    Expected: '{exp}' | Similarity: {sim:.3f}")
            print(f"    Got chunk: '{chunk}...'")
    else:
        print(f"\n  ✓ No misses at best config!")