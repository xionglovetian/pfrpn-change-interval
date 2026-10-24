import json, ssl, urllib.request, urllib.parse, time
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
for doi in ["10.1038/s41598-021-94190-9","10.3390/rs17020213","10.3390/rs14092237","10.1080/01431161.2019.1701725"]:
    print("="*100)
    try:
        d=get("https://api.openalex.org/works/https://doi.org/"+doi)
        print("TITLE:",d.get("display_name")); print("YEAR :",d.get("publication_year"))
        print("VENUE:",((d.get("primary_location") or {}).get("source") or {}).get("display_name"))
        print("ABS  :",unf(d.get("abstract_inverted_index"))[:1400])
    except Exception as e: print("FAIL",doi,e)
    time.sleep(1)

