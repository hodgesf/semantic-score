"""Inference for one LLM judge over the expert items and the audit sample.

The SYSTEM and USER prompts are the ones recorded in PROTOCOL.md. Reads data/expert_items.csv
and data/audit_manifest.csv and appends to raw/<judge>.jsonl (one line per concept: item key,
raw text, parsed rating); already-written items are skipped, so the run is resumable.
Usage: python infer.py <judge>, with judge in {gemma, nemo, phi}.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import LLM_PANEL  # noqa: E402

DATA = LLM_PANEL / "data"
RAW = LLM_PANEL / "raw"
JUDGES = {
    "gemma": "RedHatAI/gemma-3-12b-it-quantized.w4a16",
    "nemo": "RedHatAI/Mistral-Nemo-Instruct-2407-quantized.w4a16",
    "phi": "RedHatAI/phi-4-quantized.w4a16",
}
SYSTEM = (
    "You are rating biomedical concepts for GENERICNESS. A concept is GENERIC when it names a "
    "class, group, category, role or umbrella of things and would be too general to be a useful "
    'standalone answer to a biomedical question such as "what treats disease X?" or "what genes '
    'affect Y?". A concept is SPECIFIC when it names one concrete, identifiable entity (a '
    "particular drug, gene, protein, disease, process, structure, organism, product or compound) "
    "that would be a concrete, useful answer. Judge only genericness/specificity of the concept's "
    "meaning. Do not use tools, search or retrieval; judge from the name, category and description "
    "given. Breadth is not genericness: having many parts, subtypes or members does not make a "
    'named entity generic ("brain", "Alzheimer disease", "glycolysis" are specific). Accessions, '
    "gene symbols, database identifiers and catalogue labels denote one entity and are specific "
    "even if unfamiliar. Expand abbreviations before judging. Answer with JSON only."
)
USER = (
    "Concept name: {name}\nCategory: {category}\nDescription: {description}\n\n"
    "Rate the concept on this scale:\n"
    "1 = clearly generic (a class/group/role/umbrella; not a useful standalone answer)\n"
    "2 = mostly generic\n"
    "3 = borderline / unsure\n"
    "4 = mostly specific\n"
    "5 = clearly specific (one concrete, identifiable entity; a useful answer)\n\n"
    'Return exactly: {{"rating": <1-5 integer>, "confidence": <0-1>, '
    '"rationale": "<one short sentence>"}}'
)

Item = tuple[str, str, str, str, str]


def cat(c: object) -> str:
    """Category label without the biolink prefix, or ``(unknown)``."""
    return str(c).replace("biolink:", "") if isinstance(c, str) else "(unknown)"


def msgs(it: Item) -> list[dict[str, str]]:
    """Chat messages (system, user) for one item; description capped at 1200 characters."""
    desc = it[4] if isinstance(it[4], str) and it[4].strip() else "(none)"
    user = USER.format(name=it[2], category=cat(it[3]), description=desc[:1200])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(txt: str) -> tuple[int | None, object, object]:
    """Extract (rating, confidence, rationale) from the first JSON object in the model output."""
    m = re.search(r"\{.*?\}", txt, re.S)
    if not m:
        return None, None, None
    try:
        d = json.loads(m.group(0))
    except Exception:
        try:
            d = json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None, None, None
    r = d.get("rating")
    try:
        r = int(round(float(r)))
    except Exception:
        return None, d.get("confidence"), d.get("rationale")
    return (r if 1 <= r <= 5 else None), d.get("confidence"), d.get("rationale")


def load_items() -> list[Item]:
    """Expert items followed by audit items as (set, id, name, category, description) tuples."""
    E = pd.read_csv(DATA / "expert_items.csv")
    M = pd.read_csv(DATA / "audit_manifest.csv")
    items = [("expert", r.id, r["name"], r.category, r.description) for _, r in E.iterrows()]
    items += [("audit", r.id, r["name"], r.category, r.description) for _, r in M.iterrows()]
    return items


def done_keys(out_path: Path) -> set[tuple[str, str]]:
    """(set, id) keys already present in the output file."""
    if not out_path.exists():
        return set()
    with open(out_path) as fh:
        return {(json.loads(line)["set"], json.loads(line)["id"]) for line in fh}


def generate(llm: LLM, sp: SamplingParams, judge: str, todo: list[Item]) -> list:
    """Run the judge over the items; nemo has no chat template, so its prompt is built by hand."""
    if judge == "nemo":
        prompts = [
            f"[INST]{m[0]['content']}\n\n{m[1]['content']}[/INST]"
            for m in (msgs(it) for it in todo)
        ]
        return llm.generate(prompts, sp, use_tqdm=True)
    return llm.chat([msgs(it) for it in todo], sp, use_tqdm=True)


def run(judge: str, model: str, todo: list[Item], out_path: Path) -> None:
    """Generate ratings for the pending items, retrying empty outputs up to three times."""
    llm = LLM(model=model, gpu_memory_utilization=0.52, max_model_len=2048, dtype="auto", seed=2654)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=200, seed=2654)
    with open(out_path, "a") as fh:
        for attempt in range(3):
            if not todo:
                break
            outs = generate(llm, sp, judge, todo)
            remaining = []
            for it, o in zip(todo, outs):
                txt = o.outputs[0].text
                r, c, why = parse(txt)
                if r is None and attempt < 2 and txt.strip() == "":
                    remaining.append(it)
                    continue
                rec = {
                    "set": it[0],
                    "id": it[1],
                    "judge": judge,
                    "model": model,
                    "raw": txt,
                    "rating": r,
                    "confidence": c,
                    "rationale": why,
                    "attempt": attempt,
                }
                fh.write(json.dumps(rec) + "\n")
            todo = remaining


def main() -> None:
    """Parse the judge name, skip finished items and run inference on the rest."""
    judge = sys.argv[1]
    model = JUDGES[judge]
    items = load_items()
    out_path = RAW / f"{judge}.jsonl"
    done = done_keys(out_path)
    todo = [it for it in items if (it[0], it[1]) not in done]
    print(judge, model, "items", len(items), "todo", len(todo))
    if not todo:
        sys.exit(0)
    run(judge, model, todo, out_path)
    print("done", judge)


if __name__ == "__main__":
    main()
