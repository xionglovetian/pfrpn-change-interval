import json, ssl, urllib.parse, urllib.request, time
CTX=ssl.create_default_context(); UA="novelty-check/1.0"
def get(u,t=45):
    r=urllib.request.Request(u,headers={"User-Agent":UA,"Accept-Encoding":"identity"})
    with urllib.request.urlopen(r,timeout=t,context=CTX) as x: return json.loads(x.read().decode("utf-8","replace"))
def unf(ai):
    if not ai: return ""
    p={}
    for w,ii in ai.items():
        for i in ii: p[i]=w
    return " ".join(p[i] for i in sorted(p))
QS=["contribution of temporal information landslide detection deep learning",
    "importance of temporal features landslide mapping",
    "pre-event post-event imagery landslide detection deep learning",
    "temporal dynamics landslide detection benchmark Sen12Landslides",
    "landslide change signal purity background contamination"]
for q in QS:
    print("\n"+"="*96); print("### Q:",q)
    url=("https://api.openalex.org/works?search="+urllib.parse.quote(q)
         +"&per-page=8&select=display_name,publication_year,doi,primary_location,abstract_inverted_index")
    try: d=get(url)
    except Exception as e: print("FAIL",e); continue
    for w in d.get("results",[]):
        loc=(w.get("primary_location") or {}).get("source") or {}
        print(f"  [{w.get('publication_year')}] {w.get('display_name')}")
        print(f"      {loc.get('display_name')} | {(w.get('doi') or '').replace('https://doi.org/','')}")
        ab=unf(w.get("abstract_inverted_index"))
        if ab: print(f"      {ab[:200]}")
    time.sleep(1)

