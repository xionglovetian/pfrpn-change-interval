from pypdf import PdfReader
from docx import Document
import re
pdf_t = (PdfReader(r"C:\Users\ASUS\Desktop\PFRPN\01_论文\04_正文_v6_IEEE.pdf").pages[0].extract_text() or "")
d = Document(r"C:\Users\ASUS\Desktop\PFRPN\01_论文\CoverLetter_JSTARS.docx")
cl = " ".join(p.text for p in d.paragraphs)
readme = open(r"C:\Users\ASUS\Desktop\PFRPN\_gh_repo\pfrpn-change-interval\README.md", encoding="utf-8").read()
keys = ["The Change Interval Belongs to the Task Specification, not the Model"]
for k in keys:
    print("PDF     :", k.replace("\n"," ") in pdf_t.replace("\n"," "))
    print("CoverLtr:", k in cl)
    print("README  :", k in readme)
