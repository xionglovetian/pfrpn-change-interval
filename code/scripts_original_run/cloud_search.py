#!/usr/bin/env python3
import json, ssl, urllib.parse, urllib.request, time
CTX = ssl.create_default_context()
UA = "novelty-check/1.0 (mailto:researcher@example.org)"
def get(url, timeout=45):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read().decode("utf-8","replace"))
def unf(ai):
    if not ai: return ""
    pos={}
    for w,idxs in ai.items():
        for i in idxs: pos[i]=w
    return " ".join(pos[i] for i in sorted(pos))
def oa(q, n=12):
    url=("https://api.openalex.org/works?search="+urllib.parse.quote(q)
         +f"&per-page={n}&select=display_name,publication_year,doi,primary_location,abstract_inverted_index")
    try: d=get(url)
    except Exception as e: print("FAIL",q,e); return []
    out=[]
    for w in d.get("results",[]):
        loc=(w.get("primary_location") or {}).get("source") or {}
        out.append({"title":w.get("display_name",""),"year":w.get("publication_year"),
                    "venue":loc.get("display_name") or "","doi":(w.get("doi") or "").replace("https://doi.org/",""),
                    "abstract":unf(w.get("abstract_inverted_index"))})
    return out
Q={
"C1_cross_event":[
 "cross-event landslide detection generalization",
 "leave-one-event-out landslide detection",
 "event-disjoint split landslide detection",
 "cross-region transfer landslide detection benchmark",
 "landslide detection generalization unseen event",
 "landslide detection domain shift generalization",
 "landslide segmentation model generalization rainfall-induced landslides",
 "landslide detection transferability assessment"],
"C2_split_protocol":[
 "random split landslide detection data leakage",
 "train test split landslide detection deep learning importance",
 "spatial cross-validation landslide machine learning biased assessment",
 "landslide inventory deep learning split methodology",
 "Landslide4Sense benchmark split training validation"],
"C3_promptfree_rpn":[
 "prompt-free landslide detection",
 "prompt-free remote sensing object detection adaptation",
 "prompt-free universal region proposal network",
 "class-agnostic region proposal remote sensing detection",
 "open-vocabulary landslide detection"],
}
raw={}
for c,qs in Q.items():
    raw[c]={}
    for q in qs:
        r=oa(q); raw[c][q]=r
        print(f"[{c}] {q} -> {len(r)}")
        time.sleep(1.0)
json.dump(raw, open("/root/autodl-tmp/novelty_oa.json","w"), ensure_ascii=False, indent=1)
print("saved /root/autodl-tmp/novelty_oa.json")

