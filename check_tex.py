import re, os
tex = open('main.tex', encoding='utf-8').read()

# 1) cite keys vs bibitems
cites = set()
for m in re.finditer(r'\\cite\{([^}]*)\}', tex):
    for k in m.group(1).split(','):
        cites.add(k.strip())
bibs = set(re.findall(r'\\bibitem\{([^}]*)\}', tex))
print("cite keys:", len(cites), "| bibitems:", len(bibs))
missing = cites - bibs
unused = bibs - cites
print("MISSING bibitems (cited, not defined):", sorted(missing))
print("UNUSED bibitems (defined, not cited):", sorted(unused))

# 2) includegraphics files
figs = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]*)\}', tex)
for f in figs:
    ok = os.path.exists(os.path.join('figs', f)) or os.path.exists(f)
    print(f"fig {f}: {'OK' if ok else 'MISSING'}")

# 3) environment balance
begins = re.findall(r'\\begin\{([^}]*)\}', tex)
ends = re.findall(r'\\end\{([^}]*)\}', tex)
from collections import Counter
cb, ce = Counter(begins), Counter(ends)
for env in set(begins) | set(ends):
    if cb[env] != ce[env]:
        print(f"ENV IMBALANCE {env}: begin={cb[env]} end={ce[env]}")
print("env check done")

# 4) brace balance (rough)
print("brace balance {}:", tex.count('{') - tex.count('}'))
# 5) label/ref
labels = set(re.findall(r'\\label\{([^}]*)\}', tex))
refs = set(re.findall(r'\\(?:ref|eqref)\{([^}]*)\}', tex))
print("refs with no label:", sorted(refs - labels))
