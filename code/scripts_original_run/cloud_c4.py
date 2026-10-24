#!/usr/bin/env python3
import json, ssl, urllib.parse, urllib.request, time
CTX=ssl.create_default_context(); UA="novelty-check/1.0 (mailto:researcher@example.org)"
def get(u,t=45):
    r=urllib.request.Request(u,headers={"User-Agent":UA,"Accept-Encoding":"identity"})
    with urllib.request.urlopen(r,timeout=t,context=CTX) as x: return json.loads(x.read().decode("utf-8","replace"))
def unf(ai):
    if not ai: return ""
    p={}
    for w,ii in ai.items():
        for i in ii: p[i]=w
    return " ".join(p[i] for i in sorted(p))
def oa(q,n=12):
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
"C4a_change_features":[
 "temporal change features landslide detection deep learning",
 "spectral index change landslide mapping NDVI NDWI",
 "bi-temporal change landslide detection network",
 "SAR coherence change landslide detection deep learning",
 "multi-temporal landslide inventory change detection deep learning"],
"C4b_modality_importance":[
 "multimodal landslide detection modality ablation contribution",
 "quantifying modality importance multimodal remote sensing detection",
 "feature importance multi-source remote sensing landslide detection",
 "modality fusion landslide detection ablation study",
 "which modality matters landslide detection optical SAR DEM"],
"C4c_discriminative_change":[
 "change detection landslide discrimination background false positives",
 "landslide change signal specificity vs background",
 "landslide detection false positive reduction change features",
 "area under curve change features landslide detection"],
"C4d_proposal_change":[
 "region proposal change detection remote sensing",
 "class-agnostic proposal temporal change detection",
 "prompt-free proposal change detection"],
}
raw={}
for c,qs in Q.items():
    raw[c]={}
    for q in qs:
        r=oa(q); raw[c][q]=r; print(f"[{c}] {q} -> {len(r)}"); time.sleep(1.0)
json.dump(raw, open("/root/autodl-tmp/novelty_c4.json","w"), ensure_ascii=False, indent=1)
print("saved")

