"""Split the FADP PDF (Fedlex) into articles.

One passage = one article, with: number, title, chapter, text, start page.
Headers, footers and footnotes are removed when recognisable.
Limite connue : certaines notes de bas de page atypiques peuvent rester dans le texte.
"""
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ART = re.compile(r"^Art\.\s+(\d+[a-z]{0,3})(?:\s{2,}(.+))?$")
CHAPTER = re.compile(r"^(Chapitre|Section)\s+\d+[a-z]?(?:\s{2,}(.+))?$")
HEADER = re.compile(r"^(235\.1\s|Protection des données\. LF\s+235\.1)")
FOOTER = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
FOOTNOTE = re.compile(r"^\d+\s+(RS|RO|FF)\b|^\d+\s+(Nouvelle|Introduit|Abrogé|Mis à jour|Teneur|Rectifié|Erratum)")


@dataclass
class Article:
    number: str
    title: str
    chapter: str
    text: str
    page: int


def pdf_pages(pdf: Path) -> list[str]:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return out.split("\f")


def parse(pdf: Path) -> list[Article]:
    articles: list[Article] = []
    chapter, pending_chapter = "", ""
    current: dict | None = None
    for page_no, page in enumerate(pdf_pages(pdf), start=1):
        for raw in page.splitlines():
            line = raw.rstrip()
            if not line.strip() or HEADER.match(line) or FOOTER.match(line) or FOOTNOTE.match(line.strip()):
                continue
            cm = CHAPTER.match(line.strip())
            if cm:
                if cm.group(2):
                    chapter = re.sub(r"\s{2,}", " - ", line.strip())
                else:
                    pending_chapter = line.strip()
                continue
            if pending_chapter:
                chapter = f"{pending_chapter} - {line.strip()}"
                pending_chapter = ""
                continue
            m = ART.match(line.strip())
            if m:
                if current:
                    articles.append(Article(**current))
                current = {"number": m.group(1), "title": (m.group(2) or "").strip(), "chapter": chapter, "text": "", "page": page_no}
                continue
            if current is not None:
                current["text"] += line.strip() + "\n"
    if current:
        articles.append(Article(**current))
    for a in articles:
        a.text = re.sub(r"-\n(?=[a-zàâçéèêëîïôûù])", "", a.text).replace("\n", " ").strip()
    return articles


if __name__ == "__main__":
    import sys
    arts = parse(Path(sys.argv[1]))
    print(len(arts), "articles")
    for a in arts[:3] + arts[-2:]:
        print(f"Art. {a.number} [{a.title}] ({a.chapter}) p.{a.page}: {a.text[:150]}...")
    longest = max(arts, key=lambda a: len(a.text))
    print("plus long: Art.", longest.number, len(longest.text), "characters")
