"""Architecture diagram for the README, light + dark (GitHub <picture> switches them).

python scripts/make_diagrams.py   ->  docs/img/architecture-{light,dark}.svg
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"

THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", border="#d0d7de", ink="#1f2328", ink2="#59636e", faint="#e6e9ed",
                  a="#2a78d6", a_soft="#e3eefb", b="#eb6834", b_soft="#fde8de", c="#1baf7a", c_soft="#dcf4ea",
                  muted="#afb8c1"),
    "dark": dict(bg="#0d1117", card="#161b22", border="#30363d", ink="#e6edf3", ink2="#9198a1", faint="#21262d",
                 a="#3987e5", a_soft="#13263d", b="#d95926", b_soft="#3a1d12", c="#199e70", c_soft="#0f2e23",
                 muted="#484f58"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h, t, label):
        self.w, self.h, self.t, self.label, self.parts = w, h, t, label, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, color=None, weight=400, anchor="start", mono=False, italic=False):
        self.add(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{color or self.t["ink"]}" text-anchor="{anchor}"'
                 f'{" font-style=\"italic\"" if italic else ""}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=10, dash=None, sw=1.5):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke or "none"}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or self.t["ink2"]}" '
                 f'stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')

    def path(self, d, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color or self.t["ink2"]}" stroke-width="{sw}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{da}{m}/>')

    def circle(self, x, y, r, fill, stroke=None, sw=1.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def pill(self, x, y, s, fill, color, size=11.5, anchor="middle", pad=9):
        w = len(s) * size * 0.58 + 2 * pad
        x0 = x - w / 2 if anchor == "middle" else x
        self.rect(x0, y - size - 3, w, size + 10, fill, rx=(size + 10) / 2)
        self.text(x0 + w / 2, y + 1, s, size=size, color=color, weight=600, anchor="middle")

    def node(self, x, y, w, h, title, sub=None, accent=None, dash=None, mono_title=True):
        t = self.t
        fill = t[f"{accent}_soft"] if accent else t["card"]
        stroke = t[accent] if accent else t["border"]
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        cy = y + h / 2 + (-4 - 8 * (len(sub) - 1) if isinstance(sub, list) else -4 if sub else 5)
        self.text(x + w / 2, cy, title, size=14, weight=650, anchor="middle", mono=mono_title)
        if sub:
            for i, s in enumerate(sub if isinstance(sub, list) else [sub]):
                self.text(x + w / 2, cy + 19 + i * 16, s, size=12, color=t["ink2"], anchor="middle")

    def render(self):
        t = self.t
        defs = (f'<defs>'
                f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["ink2"]}"/></marker>'
                f'<marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["a"]}"/></marker>'
                f'<marker id="arb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["b"]}"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{esc(self.label)}">{defs}'
                f'<rect width="{self.w}" height="{self.h}" rx="14" fill="{t["bg"]}" stroke="{t["border"]}"/>'
                + "".join(self.parts) + "</svg>")


def header(s, title, sub):
    s.text(32, 42, title, size=19, weight=700)
    s.text(32, 64, sub, size=13, color=s.t["ink2"])


def architecture(t):
    s = SVG(1080, 600, t, "Pipeline: Lahman batting data becomes player-seasons, then for every target season "
                          "three projectors fit on prior seasons only and are scored on the next season.")
    header(s, "Rolling-origin backtest: fit on seasons up to T, score season T+1",
           "Same player pool, same seasons, three projectors. Development seasons tune; the backtest runs once.")
    # sources
    s.node(28, 120, 170, 90, "Lahman", ["Batting + People", "cdalzell/Lahman"], mono_title=False)
    s.node(28, 250, 170, 104, "data.py", ["stints merged", "PA = AB+BB+HBP+SF+SH", "baseball age (Jun 30)"])
    s.node(28, 394, 170, 80, "pytest + CI", ["validate(): no dupes,", "counts within PA"], accent="c", mono_title=False)
    s.line(113, 210, 113, 246, arrow=True)
    s.line(113, 354, 113, 390, t["c"], arrow=True)
    # fold container
    s.rect(236, 96, 590, 470, t["card"], t["border"], rx=14)
    s.text(256, 124, "for each target season T+1", size=12, weight=700, color=t["ink2"])
    s.text(806, 124, "inputs: seasons T, T-1, T-2", size=12, color=t["ink2"], anchor="end")
    s.line(198, 302, 252, 302, arrow=True)
    s.node(262, 150, 170, 80, "Last season", "rate in T", mono_title=False)
    s.node(262, 250, 170, 96, "Marcel", ["5/4/3 weights", "+1,200 PA league", "+ age factor"], mono_title=False)
    # bayes block
    s.rect(462, 146, 344, 400, t["a_soft"], t["a"], rx=14, sw=2)
    s.text(634, 172, "model.py  ·  PyMC", size=15, weight=700, anchor="middle", mono=True, color=t["a"])
    s.node(482, 188, 304, 84, "population prior", ["theta_i ~ Normal(mu + b·age, sigma)", "logit scale, per stat"],
           mono_title=False)
    s.node(482, 290, 304, 84, "weighted binomial likelihood", ["seasons T, T-1, T-2 at 1 / 0.8 / 0.6",
                                                               "offset by that season's league rate"], mono_title=False)
    s.node(482, 392, 304, 84, "projection for T+1", ["+ delta-method aging curve (train years)",
                                                     "+ talent drift tau (set on dev)"], mono_title=False)
    s.line(634, 272, 634, 286, t["a"], arrow=True, marker="ara")
    s.line(634, 374, 634, 388, t["a"], arrow=True, marker="ara")
    s.text(634, 504, "NUTS, 4 chains · R-hat and ESS logged per fit", size=12, color=t["ink2"], anchor="middle")
    s.text(634, 524, "posterior predictive with actual PA gives 80% intervals", size=12, color=t["ink2"],
           anchor="middle")
    # scoring
    s.node(866, 200, 190, 150, "evaluate.py", ["players with 200+ PA", "in T and T+1", "PA-weighted RMSE",
                                               "80% coverage"], accent="b")
    s.line(826, 275, 862, 275, t["b"], sw=2, arrow=True, marker="arb")
    s.text(347, 384, "3 projections per", size=12, color=t["ink2"], anchor="middle")
    s.text(347, 402, "player and stat", size=12, color=t["ink2"], anchor="middle")
    # protocol strip
    s.rect(866, 386, 190, 180, t["card"], t["border"], rx=12)
    s.text(961, 412, "Protocol", size=13, weight=700, anchor="middle")
    s.pill(961, 444, "2010-2014  dev", t["faint"], t["ink"], size=11.5)
    s.pill(961, 478, "2015-2025  backtest", t["b_soft"], t["b"], size=11.5)
    s.text(961, 512, "run once, logged", size=11.5, color=t["ink2"], anchor="middle")
    s.text(961, 532, "2020 skipped as target", size=11.5, color=t["ink2"], anchor="middle")
    s.text(961, 552, "(still an input)", size=11.5, color=t["ink2"], anchor="middle")
    return s.render()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        (OUT / f"architecture-{name}.svg").write_text(architecture(t))
    print("diagrams written to", OUT)
