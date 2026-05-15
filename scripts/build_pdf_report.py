"""
analysis_report.md → PDF 변환 스크립트
Chrome headless 사용 (한글 완전 지원)
"""

import subprocess
import sys
from pathlib import Path
import markdown
import re

ROOT       = Path(__file__).parent.parent
MD_PATH    = ROOT / 'reports' / 'analysis_report.md'
HTML_PATH  = ROOT / 'reports' / 'analysis_report.html'
PDF_PATH   = ROOT / 'reports' / 'analysis_report.pdf'

# ── Markdown → HTML ───────────────────────────────────────────
md_text = MD_PATH.read_text(encoding='utf-8')

html_body = markdown.markdown(
    md_text,
    extensions=['tables', 'fenced_code', 'toc', 'nl2br'],
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

* { box-sizing: border-box; }

body {
    font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    font-size: 11pt;
    line-height: 1.7;
    color: #1a1a1a;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px 40px 40px 40px;
}

h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #1a3a5c;
    border-bottom: 3px solid #1a3a5c;
    padding-bottom: 10px;
    margin-top: 30px;
}

h2 {
    font-size: 15pt;
    font-weight: 700;
    color: #1a3a5c;
    border-bottom: 2px solid #3a7abf;
    padding-bottom: 6px;
    margin-top: 30px;
}

h3 {
    font-size: 12pt;
    font-weight: 700;
    color: #2c5282;
    margin-top: 22px;
    margin-bottom: 8px;
}

h4 {
    font-size: 11pt;
    font-weight: 700;
    color: #2d3748;
    margin-top: 18px;
    margin-bottom: 6px;
}

p { margin: 6px 0 10px 0; }

/* 표 */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 10pt;
}

th {
    background-color: #2b6cb0;
    color: white;
    padding: 8px 10px;
    text-align: left;
    font-weight: 500;
}

td {
    padding: 6px 10px;
    border: 1px solid #cbd5e0;
}

tr:nth-child(even) td { background-color: #f7fafc; }
tr:hover td { background-color: #ebf4ff; }

/* 코드 */
code {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    background-color: #f1f5f9;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 9.5pt;
    color: #c7254e;
}

pre {
    background-color: #1e2433;
    color: #e2e8f0;
    padding: 14px 16px;
    border-radius: 6px;
    overflow-x: auto;
    font-size: 9pt;
    line-height: 1.5;
    margin: 12px 0;
}

pre code {
    background: none;
    color: #e2e8f0;
    padding: 0;
}

/* 인용 */
blockquote {
    border-left: 4px solid #3a7abf;
    background-color: #eff8ff;
    margin: 12px 0;
    padding: 8px 16px;
    color: #2c5282;
    border-radius: 0 4px 4px 0;
    font-size: 10.5pt;
}

blockquote p { margin: 4px 0; }

/* 리스트 */
ul, ol {
    padding-left: 22px;
    margin: 8px 0;
}

li { margin-bottom: 4px; }

/* 구분선 */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 24px 0;
}

/* 강조 */
strong { color: #1a3a5c; }

/* 페이지 나누기 방지 */
h2, h3 { page-break-after: avoid; }
table   { page-break-inside: avoid; }

/* 페이지 헤더/푸터 */
@page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #718096;
    }
}
"""

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>한국남동발전 대기오염물질 배출 최적화 분석 보고서</title>
<style>
{CSS}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

HTML_PATH.write_text(HTML_TEMPLATE, encoding='utf-8')
print(f'[OK] HTML 생성: {HTML_PATH}')

# ── HTML → PDF (Chrome headless) ─────────────────────────────
chrome_candidates = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    'google-chrome',
    'chromium',
]

chrome_bin = None
for c in chrome_candidates:
    result = subprocess.run(['test', '-f', c], capture_output=True)
    if result.returncode == 0 or Path(c).exists():
        chrome_bin = c
        break

if chrome_bin is None:
    print('[ERROR] Chrome/Chromium을 찾을 수 없습니다.')
    sys.exit(1)

cmd = [
    chrome_bin,
    '--headless',
    '--disable-gpu',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    f'--print-to-pdf={PDF_PATH}',
    '--print-to-pdf-no-header',
    '--run-all-compositor-stages-before-draw',
    '--virtual-time-budget=5000',
    f'file://{HTML_PATH}',
]

print(f'[실행] Chrome headless PDF 변환...')
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0 and PDF_PATH.exists():
    size_kb = PDF_PATH.stat().st_size // 1024
    print(f'[OK] PDF 생성 완료: {PDF_PATH}  ({size_kb} KB)')
else:
    print(f'[ERROR] 변환 실패')
    print(result.stderr[:500] if result.stderr else '(오류 없음)')
    sys.exit(1)
