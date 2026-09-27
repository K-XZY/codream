"""M5, on the Mac: put exported results into the results page.

usage: python tools/fill_page.py EXPORT_DIR PAGE_HTML
Replaces the contents of the page's <script id="codream-data"> block with EXPORT_DIR/codream-data.json
and copies EXPORT_DIR/codream-exp/ next to the page. Touches nothing else in the page.
"""
import json, os, re, shutil, sys

src, page = sys.argv[1], sys.argv[2]
data = json.load(open(os.path.join(src, "codream-data.json")))
html = open(page).read()
pat = re.compile(r'(<script type="application/json" id="codream-data">\n).*?(\n</script>)', re.S)
assert len(pat.findall(html)) == 1, "expected exactly one codream-data block"
html = pat.sub(lambda m: m.group(1) + json.dumps(data, separators=(",", ":")) + m.group(2), html)
open(page, "w").write(html)
dst = os.path.join(os.path.dirname(os.path.abspath(page)), "codream-exp")
shutil.copytree(os.path.join(src, "codream-exp"), dst, dirs_exist_ok=True)
print("filled", page, "and copied images to", dst)
