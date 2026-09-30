from __future__ import annotations

import re

from bs4 import BeautifulSoup
from markdownify import markdownify as to_markdown

REMOVABLE = (
    "script, style, noscript, nav, footer, header, form, iframe, svg, "
    "button, [role='navigation'], [aria-label*='cookie' i], "
    "[class*='cookie' i], [id*='cookie' i], [class*='breadcrumb' i], "
    "[class*='recommend' i], [class*='related' i], [class*='modal' i]"
)


def extract_markdown(html: str, title: str) -> str:
    """Extract the page's main content and convert it without LLM rewriting."""
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select(REMOVABLE):
        node.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
    )
    if main is None:
        raise ValueError("No HTML body or main content found")

    markdown = to_markdown(
        str(main),
        heading_style="ATX",
        bullets="-",
        strip=["img", "picture", "source"],
    )
    markdown = re.sub(r"\n[ \t]+\n", "\n\n", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown).strip()
    if not markdown:
        raise ValueError("Page extraction produced no text")
    if not markdown.startswith("# "):
        markdown = f"# {title}\n\n{markdown}"
    return markdown
