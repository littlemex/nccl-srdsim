# SPDX-License-Identifier: Apache-2.0
"""A small SVG builder for the figures in docs/, in English and Japanese.

Every figure script in this directory defines NAME and draw(lang) -> Fig; build.py writes
docs/images/NAME.svg (English) and docs/images/NAME.ja.svg (Japanese). Roles are told apart by colour
and by a pattern, so colour is never the only cue. Labels are measured roughly when they are drawn,
and build.py fails when one would not fit the box or the canvas it is drawn in.
"""
import html

FONT = ("Helvetica, Arial, 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Noto Sans CJK JP', "
        "'Noto Sans JP', sans-serif")
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
INK, MUTED, SOFT = "#222222", "#555555", "#F5F5F5"

# Okabe-Ito colours, which stay apart for the common forms of colour blindness, each with its pattern.
ROLE = {
    "nccl":    ("#0072B2", "cross"),         # NCCL and the application above it
    "sim":     ("#009E73", "diagonal"),      # this plugin
    "wire":    ("#E69F00", "grid"),          # the transport between ranks
    "target":  ("#CC79A7", "dots"),          # the target rank's memory
    "sig":     ("#D55E00", "backdiagonal"),  # signals
    "neutral": ("#888888", None),
}


class Lang:
    def __init__(self, code):
        self.code = code

    def t(self, en, ja):
        return ja if self.code == "ja" else en


def text_width(s, size, mono=False):
    """A rough width in pixels: CJK characters are about one em wide, Latin ones about 0.56 em."""
    w = 0.0
    for ch in s:
        if ord(ch) >= 0x2E80:
            w += 1.0
        elif mono:
            w += 0.6
        elif ch in "il.,:;'|!":
            w += 0.3
        elif ch.isupper() or ch in "mwMW@%":
            w += 0.72
        else:
            w += 0.56
    return w * size


class Fig:
    def __init__(self, w, h, lang, title=None):
        self.w, self.h, self.lang = w, h, lang
        self.items, self.defs, self.problems, self.rects = [], {}, [], []
        self.items.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#FFFFFF"/>')
        if title:
            self.text(w / 2, 30, title, size=19, weight="bold")

    # -- text -------------------------------------------------------------------------------
    def _check(self, s, size, x, anchor, limit, mono, where):
        width = text_width(s, size, mono)
        left = x - (width / 2 if anchor == "middle" else width if anchor == "end" else 0)
        lo, hi = limit if limit else (0, self.w)
        if left < lo - 1 or left + width > hi + 1:
            self.problems.append(f"{where}: '{s}' is about {width:.0f}px wide and does not fit {lo:.0f}..{hi:.0f}")

    def text(self, x, y, s, size=14, weight="normal", anchor="middle", color=None, italic=False, mono=False,
             limit=None):
        if isinstance(s, (list, tuple)):
            for i, line in enumerate(s):
                self.text(x, y + i * size * 1.3, line, size, weight, anchor, color, italic, mono, limit)
            return
        self._check(s, size, x, anchor, limit, mono, "text")
        st = ' font-style="italic"' if italic else ""
        fam = MONO if mono else FONT
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" '
                          f'font-weight="{weight}" text-anchor="{anchor}" fill="{color or INK}"{st}>'
                          f'{html.escape(s)}</text>')

    def _place(self, x, y, w, h, what):
        """Record a box; report it when it leaves the canvas or overlaps another box."""
        if x < 0 or y < 0 or x + w > self.w + 0.5 or y + h > self.h + 0.5:
            self.problems.append(f"{what}: {x:.0f},{y:.0f} {w:.0f}x{h:.0f} leaves the {self.w}x{self.h} canvas")
        for (ox, oy, ow, oh, other) in self.rects:
            if x < ox + ow - 0.5 and ox < x + w - 0.5 and y < oy + oh - 0.5 and oy < y + h - 0.5:
                self.problems.append(f"{what} overlaps {other}")
        self.rects.append((x, y, w, h, what))

    # -- patterns ---------------------------------------------------------------------------
    def _pattern(self, role):
        colour, kind = ROLE[role]
        if kind is None:
            return None
        pid = f"p-{role}"
        if pid not in self.defs:
            body = {
                "diagonal": '<path d="M-2,2 l4,-4 M0,10 l10,-10 M8,12 l4,-4" stroke-width="1.2"/>',
                "backdiagonal": '<path d="M-2,8 l4,4 M0,0 l10,10 M8,-2 l4,4" stroke-width="1.2"/>',
                "cross": '<path d="M0,5 h10 M5,0 v10" stroke-width="0.9"/>',
                "grid": '<path d="M0,0.5 h10 M0.5,0 v10" stroke-width="1.0"/>',
                "dots": '<circle cx="5" cy="5" r="1.4" stroke="none" fill="{c}"/>',
            }[kind].replace("{c}", colour)
            self.defs[pid] = (f'<pattern id="{pid}" width="10" height="10" patternUnits="userSpaceOnUse">'
                              f'<g stroke="{colour}" opacity="0.35">{body}</g></pattern>')
        return pid

    # -- shapes -----------------------------------------------------------------------------
    def box(self, x, y, w, h, label, role, sub=None, dash=False, size=15, sub_size=12, pattern=True):
        """A component: role colour on the left edge and the outline, the role's pattern as fill."""
        colour = ROLE[role][0]
        self._place(x, y, w, h, f"box '{label if isinstance(label, str) else label[0]}'")
        d = ' stroke-dasharray="7 4"' if dash else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="#FFFFFF" '
                          f'stroke="{colour}" stroke-width="2.4"{d}/>')
        pid = self._pattern(role) if pattern else None
        if pid:
            self.items.append(f'<rect x="{x+10}" y="{y+1.2}" width="{w-11.2}" height="{h-2.4}" rx="8" '
                              f'fill="url(#{pid})"/>')
        self.items.append(f'<rect x="{x}" y="{y}" width="9" height="{h}" rx="3" fill="{colour}"/>')
        lim = (x + 12, x + w - 4)
        subs = sub if isinstance(sub, (list, tuple)) else ([sub] if sub else [])
        lines = label if isinstance(label, (list, tuple)) else [label]
        total = len(lines) * size * 1.25 + len(subs) * sub_size * 1.3
        top = y + h / 2 - total / 2 + size * 0.95
        cx = x + w / 2 + 5
        for i, line in enumerate(lines):
            self._halo_text(cx, top + i * size * 1.25, line, size, "bold", lim)
        for j, line in enumerate(subs):
            self._halo_text(cx, top + len(lines) * size * 1.25 + j * sub_size * 1.3 - 2, line, sub_size, "normal",
                            lim, colour=MUTED)
        if total > h - 4:
            self.problems.append(f"box '{lines[0]}': {len(lines)+len(subs)} lines do not fit {h}px")

    def _halo_text(self, x, y, s, size, weight, limit, colour=None):
        """Text with a white outline so the pattern under it never hurts legibility."""
        self._check(s, size, x, "middle", limit, False, "box")
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
                          f'font-weight="{weight}" text-anchor="middle" fill="{colour or INK}" '
                          f'stroke="#FFFFFF" stroke-width="3.5" paint-order="stroke">{html.escape(s)}</text>')

    def group(self, x, y, w, h, label, role, dash=False):
        colour = ROLE[role][0]
        d = ' stroke-dasharray="7 4"' if dash else ""
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="13" fill="{SOFT}" '
                          f'stroke="{colour}" stroke-width="1.6"{d}/>')
        self.text(x + 14, y + 22, label, size=14, weight="bold", anchor="start", color=colour,
                  limit=(x, x + w))

    def cell(self, x, y, w, h, label, role, size=14, mono=False, strike=False):
        """A unit, a chunk or a field: tinted fill, the role's pattern, a bold label."""
        colour = ROLE[role][0]
        self._place(x, y, w, h, f"cell '{label if isinstance(label, str) else label[0]}' at {x:.0f},{y:.0f}")
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{colour}" '
                          f'fill-opacity="0.16" stroke="{colour}" stroke-width="2"/>')
        pid = self._pattern(role)
        if pid:
            self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="url(#{pid})"/>')
        if strike:
            self.items.append(f'<line x1="{x+4}" y1="{y+h-4}" x2="{x+w-4}" y2="{y+4}" stroke="{INK}" '
                              f'stroke-width="2.5"/>')
        lines = label if isinstance(label, (list, tuple)) else [label]
        top = y + h / 2 - (len(lines) - 1) * size * 0.65 + size * 0.35
        for i, line in enumerate(lines):
            self._check(line, size, x + w / 2, "middle", (x + 2, x + w - 2), mono, "cell")
            fam = MONO if mono else FONT
            self.items.append(f'<text x="{x+w/2:.1f}" y="{top + i*size*1.3:.1f}" font-family="{fam}" '
                              f'font-size="{size}" font-weight="bold" text-anchor="middle" fill="{INK}" '
                              f'stroke="#FFFFFF" stroke-width="3" paint-order="stroke">{html.escape(line)}</text>')

    def arrow(self, x1, y1, x2, y2, label=None, colour=None, dash=False, lx=None, ly=None, anchor="middle",
              width=2.0, size=12, both=False):
        col = colour or INK
        d = ' stroke-dasharray="7 4"' if dash else ""
        mid = f"m{len(self.items)}"
        self.defs[mid] = (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                          f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
                          f'fill="{col}"/></marker>')
        start = f' marker-start="url(#{mid})"' if both else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="{width}"'
                          f'{d} marker-end="url(#{mid})"{start}/>')
        if label:
            self.text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 7,
                      label, size=size, color="#333333", anchor=anchor)

    def path(self, d, colour=None, width=2.0, dash=False, arrow=True):
        col = colour or INK
        mid = f"m{len(self.items)}"
        self.defs[mid] = (f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                          f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
        dd = ' stroke-dasharray="7 4"' if dash else ""
        me = f' marker-end="url(#{mid})"' if arrow else ""
        self.items.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{width}"{dd}{me}/>')

    def step(self, x, y, n, r=13):
        """A numbered badge; the numbers match the numbered steps in the text."""
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{INK}"/>')
        self.items.append(f'<text x="{x}" y="{y+5}" font-family="{FONT}" font-size="14" font-weight="bold" '
                          f'text-anchor="middle" fill="#FFFFFF">{n}</text>')

    def line(self, x1, y1, x2, y2, colour=None, width=1.5, dash=False):
        d = ' stroke-dasharray="5 4"' if dash else ""
        self.items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour or MUTED}" '
                          f'stroke-width="{width}"{d}/>')

    def mark(self, x, y, ok, size=16):
        """A pass or fail mark drawn as a shape, not only as a colour."""
        if ok:
            self.items.append(f'<path d="M{x-7},{y} l5,6 l10,-12" fill="none" stroke="#00802F" stroke-width="3"/>')
        else:
            self.items.append(f'<path d="M{x-6},{y-6} l12,12 M{x+6},{y-6} l-12,12" stroke="#D55E00" '
                              f'stroke-width="3"/>')

    def legend(self, x, y, roles, labels):
        for i, (role, lab) in enumerate(zip(roles, labels)):
            yy = y + i * 24
            self.cell(x, yy, 26, 16, "", role)
            self.text(x + 34, yy + 13, lab, size=12, anchor="start", color="#333333")

    def svg(self):
        defs = "<defs>" + "".join(self.defs.values()) + "</defs>" if self.defs else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}">' + defs + "".join(self.items) + "</svg>\n")
