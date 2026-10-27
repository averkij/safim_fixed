"""Codeforces catalog, snapshot history, and explicit source acquisition."""
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

from .storage import canonical, digest, read_json, read_rows, write_json


def epoch(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone: " + value)
    return int(parsed.timestamp())


def iso(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


class Collector:
    def __init__(self, store, refresh=False):
        self.store = store
        self.refresh = refresh
        self.last_request = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "safim-v3-research/0.1"

    def fetch(self, url):
        index = self.store.root / "cache" / (digest(url) + ".json")
        if index.exists() and not self.refresh:
            snap = read_json(index)
            data = (self.store.root / snap["raw_path"]).read_bytes()
            if digest(data) != snap["sha256"]:
                raise ValueError("Cached response checksum mismatch: " + url)
            return data, snap
        time.sleep(max(0, 2.1 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        try:
            response = self.session.get(url, timeout=30)
        except requests.RequestException as exc:
            self.store.snapshot(canonical({"error": str(exc)}), url, status=0)
            raise
        snap = self.store.snapshot(response.content, url, response.status_code)
        response.raise_for_status()
        write_json(index, snap)
        return response.content, snap

    def api(self, method, **params):
        import json
        data, snap = self.fetch("https://codeforces.com/api/" + method + "?" + urlencode(sorted(params.items())))
        result = json.loads(data)
        if result.get("status") != "OK":
            raise ValueError(f"Codeforces API failure: {result.get('comment')}")
        return result["result"], snap


def inventory(store, config, refresh=False):
    collector = Collector(store, refresh)
    contests, cs = collector.api("contest.list", gym="false")
    catalog, ps = collector.api("problemset.problems", lang="en")
    start, end = epoch(config["start"]), epoch(config["end"])
    selected = {c["id"]: c for c in contests if c.get("phase") == "FINISHED"
                and start <= c.get("startTimeSeconds", 0)
                and c.get("startTimeSeconds", 0) + c.get("durationSeconds", 0) < end}
    contest_rows = []
    for c in sorted(selected.values(), key=lambda c: c["id"]):
        divisions = re.findall(r"Div\.\s*(\d+)", c["name"])
        contest_rows.append({**c, "contest_id": c["id"], "snapshot_id": cs["snapshot_id"],
            "start_at": iso(c["startTimeSeconds"]), "end_at": iso(c["startTimeSeconds"] + c["durationSeconds"]),
            "divisions_inferred": divisions or None, "division_derivation": "contest-name regex",
            "source_url": f"https://codeforces.com/contest/{c['id']}"})
    problems = []
    for p in catalog["problems"]:
        if p.get("contestId") not in selected:
            continue
        c = selected[p["contestId"]]
        excluded = ["interactive"] if "interactive" in p.get("tags", []) else []
        problems.append({**p, "problem_id": f"{p['contestId']}{p['index']}", "contest_id": p["contestId"],
            "snapshot_id": ps["snapshot_id"], "rating": p.get("rating"), "tags": p.get("tags"),
            "points": p.get("points"), "contest_date": iso(c["startTimeSeconds"]),
            "published_at": None, "publication_evidence": None, "publication_precision": None,
            "earlier_appearances": None, "family_id": None,
            "time_limit": None, "memory_limit": None, "statement": None,
            "contest_metadata": {**c, "snapshot_id": cs["snapshot_id"]},
            "source_url": f"https://codeforces.com/contest/{p['contestId']}/problem/{p['index']}",
            "eligibility": "excluded" if excluded else "pending_review", "exclusion_reasons": excluded})
    problems.sort(key=lambda p: p["problem_id"])
    return store.stage("inventory", [cs["snapshot_id"], ps["snapshot_id"]], config,
                       {"contests": contest_rows, "problems": problems})


def select_pilot(problems, count, seed):
    """Round-robin month/rating strata, deterministic and problem-unique."""
    groups = {}
    for p in problems:
        if p["eligibility"] == "excluded":
            continue
        rating = p.get("rating")
        key = (p["contest_date"][:7], "unknown" if rating is None else str(int(rating) // 800))
        groups.setdefault(key, []).append(p)
    for rows in groups.values():
        rows.sort(key=lambda p: digest([seed, p["problem_id"]]))
    result = []
    while groups and len(result) < count:
        for key in sorted(list(groups)):
            if len(result) == count:
                break
            result.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
    return result


def language(label):
    if "C++" in label:
        return "cpp"
    if "Python" in label or "PyPy" in label:
        return "python"
    if "Java" in label and "JavaScript" not in label:
        return "java"
    if "C#" in label:
        return "csharp"
    return None


def parse_statement(html):
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".problem-statement")
    if root is None:
        raise ValueError("Problem statement absent (possibly an access challenge)")
    def content(selector):
        node = root.select_one(selector)
        return node.get_text("\n", strip=True) if node else None
    tests = []
    inputs, outputs = root.select(".sample-test .input pre"), root.select(".sample-test .output pre")
    if len(inputs) != len(outputs):
        raise ValueError("Unpaired sample tests")
    for a, b in zip(inputs, outputs):
        tests.append({"input": a.get_text("\n"), "output": [b.get_text("\n")], "origin": "sample"})
    return {"statement": root.get_text("\n", strip=True), "input_specification": content(".input-specification"),
            "output_specification": content(".output-specification"), "time_limit": content(".time-limit"),
            "memory_limit": content(".memory-limit"), "samples": tests}


def parse_source(html):
    node = BeautifulSoup(html, "html.parser").select_one("#program-source-text")
    if node is None:
        raise ValueError("Submission source absent (possibly an access challenge)")
    lines = node.select("ol.linenums > li")
    if lines:
        # Browser prettification wraps lines in list items and inserts NBSP for
        # blank lines. Preserve token spacing within each line, not between spans.
        return "\n".join("" if line.get_text() == "\xa0" else line.get_text() for line in lines)
    return node.get_text()


def acquire(store, config, inventory_dir, count=20, pages=2, refresh=False):
    """Bounded pilot acquisition. Partial failures are explicit output rows."""
    collector = Collector(store, refresh)
    problems = select_pilot(read_rows(inventory_dir / "problems.jsonl"), count, config["seed"])
    submissions, failures = [], []
    statuses = {}
    for p in problems:
        try:
            data, snap = collector.fetch(p["source_url"])
            p.update(parse_statement(data))
            p["statement_snapshot_id"] = snap["snapshot_id"]
        except (requests.RequestException, ValueError) as exc:
            failures.append({"problem_id": p["problem_id"], "stage": "statement", "error": str(exc)})
            # One access denial is evidence to stop page acquisition, not hammer the site.
            if isinstance(exc, requests.HTTPError) and exc.response.status_code in (401, 403, 429):
                break
            continue
        cid = p["contest_id"]
        if cid not in statuses:
            statuses[cid] = []
            for page in range(pages):
                batch, snap = collector.api("contest.status", contestId=cid, **{"from": page * 1000 + 1, "count": 1000})
                statuses[cid].extend((s, snap) for s in batch)
                if len(batch) < 1000:
                    break
        counts = {}
        for s, snap in sorted(statuses[cid], key=lambda pair: pair[0]["id"]):
            lang = language(s["programmingLanguage"])
            if (s["problem"]["index"] != p["index"] or s.get("verdict") != "OK" or lang is None
                    or not epoch(config["start"]) <= s["creationTimeSeconds"] < epoch(config["end"])
                    or counts.get(lang, 0) >= config["max_solutions_per_problem_language"]):
                continue
            url = f"https://codeforces.com/contest/{cid}/submission/{s['id']}"
            try:
                raw, source_snap = collector.fetch(url)
                source = parse_source(raw)
                submissions.append({"submission_id": s["id"], "problem_id": p["problem_id"], "lang": lang,
                    "submitted_at": iso(s["creationTimeSeconds"]), "language_label": s["programmingLanguage"],
                    "verdict": s["verdict"], "passed_test_count": s.get("passedTestCount"),
                    "time_ms": s.get("timeConsumedMillis"), "memory_bytes": s.get("memoryConsumedBytes"),
                    "author_key": digest(s.get("author", {}).get("members", [])), "source": source,
                    "source_sha256": digest(source.encode("utf-8")), "source_url": url,
                    "snapshot_id": snap["snapshot_id"], "source_snapshot_id": source_snap["snapshot_id"]})
                counts[lang] = counts.get(lang, 0) + 1
            except (requests.RequestException, ValueError) as exc:
                failures.append({"submission_id": s["id"], "stage": "source", "error": str(exc)})
                if isinstance(exc, requests.HTTPError) and exc.response.status_code in (401, 403, 429):
                    return store.stage("acquire", [str(inventory_dir)], config,
                        {"problems": problems, "submissions": submissions, "failures": failures}, {"pages_per_contest": pages})
    return store.stage("acquire", [str(inventory_dir)], config,
        {"problems": problems, "submissions": submissions, "failures": failures}, {"pages_per_contest": pages})
