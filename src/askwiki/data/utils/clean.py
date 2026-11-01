import re

TAIL_SECTIONS = re.compile(
    r'\n\s*(See also|References|External links|Further reading|Notes|Footnotes|'
    r'Citations|Sources|Bibliography|Notes and references)\s*\n.*', re.S
)
EMPTY_PARENS = re.compile(r'\(\s*(?:[;,]\s*)*\)')
LEAD_PUNCT_PAREN = re.compile(r'\(\s*(?:[;,]\s*)+')
TRAIL_PUNCT_PAREN = re.compile(r'(?:\s*[;,])+\s*\)')
MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December'
GLUED_DATE = re.compile(rf'(\d)({MONTHS})')
TEMPLATES = re.compile(r'\{\{[^{}]*\}\}')
HTML = re.compile(r'<ref[^>]*/>|<ref[^>]*>.*?</ref>|<[^>]+>', re.S)
TABLE_ROWS = re.compile(r'^\s*(\{\||\|\}|\|-|\|).*$', re.M)
MULTISPACE = re.compile(r'[ \t]{2,}')
SPACE_PUNCT = re.compile(r' +([,.;:])')
BLANK_LINES = re.compile(r'\n{3,}')


def clean_text(text: str) -> str:
    """
    claude generated text cleaning function after it went through the wikipedia data.
    general cleaning of scraped wikipedia pages - some html only things like the 
    pronunciation of words, etc. are ommitted and thus leave things like empty
    parentheses; as well, removing tail sections like citations, sources and further
    linked references.
    """
    text = TAIL_SECTIONS.sub('', text)
    text = HTML.sub('', text)
    for _ in range(3):
        text = TEMPLATES.sub('', text)
    text = TABLE_ROWS.sub('', text)
    text = text.replace("'''", '').replace("''", '')
    text = LEAD_PUNCT_PAREN.sub('(', text)
    text = TRAIL_PUNCT_PAREN.sub(')', text)
    text = EMPTY_PARENS.sub('', text)
    text = GLUED_DATE.sub(r'\1 – \2', text)
    text = MULTISPACE.sub(' ', text)
    text = SPACE_PUNCT.sub(r'\1', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = BLANK_LINES.sub('\n\n', text)
    return text.strip()