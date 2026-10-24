import json, ssl, urllib.request, time
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
DOIS=["10.1038/s41598-025-31208-6","10.1371/journal.pone.0347324","10.1038/s41598-026-59282-4",
      "10.3390/rs18081115","10.1038/s41597-025-06167-2","10.3390/rs15133340","10.48550/arxiv.2606.18661"]
import sys
for doi in DOIS:
    print("="*98)
    try:
        d=get("https://api.openalex.org/works/https://doi.org/"+doi)
        print("TITLE:",d.get("display_name")); print("YEAR :",d.get("publication_year"),
              "| VENUE:",((d.get("primary_location") or {}).get("source") or {}).get("display_name"))
        print("ABS  :",unf(d.get("abstract_inverted_index"))[:1100])
    except Exception as e: print("FAIL",doi,repr(e)[:120])
    time.sleep(0.8)

