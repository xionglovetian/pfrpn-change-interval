#!/usr/bin/env python3
import json, ssl, urllib.parse, urllib.request, time, re
CTX = ssl.create_default_context()
UA = "novelty-check/1.0 (mailto:researcher@example.org)"

def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def unf(ai):
    if not ai: return ""
    pos = {}
    for w, idxs in ai.items():
        for i in idxs: pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))

def oa(q, n=10):
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(q)
           + f"&per-page={n}&select=display_name,publication_year,doi,primary_location,abstract_inverted_index")
    try:
        d = get(url)
    except Exception as e:
        print("  FAIL", q, e); return []
    out=[]
    for w in d.get("results",[]):
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append({"t": w.get("display_name",""), "y": w.get("publication_year"),
                    "v": (loc.get("display_name") or ""), "doi": (w.get("doi") or "").replace("https://doi.org/",""),
                    "ab": unf(w.get("abstract_inverted_index"))})
    return out

QUERIES = [
 "cross-event landslide detection generalization",
 "leave-one-event-out landslide detection",
 "event-disjoint split landslide detection deep learning",
 "cross-region transfer landslide detection benchmark",
 "landslide detection generalization unseen events",
 "same-protocol benchmark landslide detection transferability",
 "landslide object detection dataset split random event",
]
print("################ OPENALEX SEARCH")
for q in QUERIES:
    print("\n" + "="*100); print("### Q:", q)
    for r in oa(q, 8):
        print(f"  [{r['y']}] {r['t']}")
        print(f"      {r['v'][:80]} | {r['doi']}")
        if r["ab"]: print(f"      ABS: {r['ab'][:330]}")
    time.sleep(1.0)

print("\n\n################ DOI DETAIL")
for doi in ["10.1080/15481603.2026.2647162", "10.1109/tcsvt.2026.3686175"]:
    print("\n" + "="*100); print("### DOI:", doi)
    try:
        d = get("https://api.openalex.org/works/https://doi.org/"+doi)
        print("TITLE:", d.get("display_name"))
        print("YEAR :", d.get("publication_year"))
        print("ABS  :", unf(d.get("abstract_inverted_index"))[:1600] or "(none)")
    except Exception as e:
        print("  FAIL", e)
    time.sleep(1.0)

