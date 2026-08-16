"""Diagramas em SVG desenhados à mão.

Por que SVG e não Plotly: o loop de governança é um desenho, não um gráfico de
dados. Formas do Plotly forçam eixos, hover e uma tipografia que não se controla
bem — o resultado parece rascunho. Com SVG dá para ter cantos arredondados,
curvas de Bézier, sombra, hierarquia tipográfica real e escala fluida, e ainda
assim acompanhar o tema claro/escuro pelos tokens do app.
"""

from __future__ import annotations

from src.utils.styling import BLUE, GOLD, NAVY, TEAL, tokens

# (título, subtítulo, ícone, cor)
LOOP_NODES = [
    ("MMM", "Estratégico · top-down", "🏛️", NAVY),
    ("MTA", "Tático · bottom-up", "🔬", TEAL),
    ("Testes A/B", "Validação causal", "🧪", GOLD),
    ("Learning Repository", "Governança", "🔁", BLUE),
]

LOOP_EDGES = [
    "aponta onde investir",
    "gera hipóteses",
    "registra o aprendizado",
    "recalibra o modelo",
]


def governance_loop_svg(height: int = 560) -> str:
    """Loop MMM ⇄ MTA ⇄ Testes A/B ⇄ Learning Repository."""
    t = tokens()
    dark = t.name == "dark"

    card = t.surface
    card_stroke = t.border
    text = t.text
    muted = t.text_muted
    shadow_opacity = 0.45 if dark else 0.16
    glow = 0.18 if dark else 0.10

    # Geometria: quatro cards nas pontas de um losango, com folga entre eles.
    cards = [
        (500, 104),   # topo — MMM
        (812, 322),   # direita — MTA
        (500, 540),   # base — Testes A/B
        (188, 322),   # esquerda — Learning Repository
    ]
    w, h, r = 236, 96, 18

    # Curvas entre as bordas dos cards, no sentido horário.
    edges = [
        # (x1, y1, cx, cy, x2, y2, rótulo x, rótulo y, âncora)
        (622, 128, 752, 150, 782, 268, 748, 168, "start"),
        (782, 376, 752, 496, 622, 518, 748, 486, "start"),
        (378, 518, 248, 496, 218, 376, 252, 486, "end"),
        (218, 268, 248, 150, 378, 128, 252, 168, "end"),
    ]

    parts: list[str] = [
        f'<svg viewBox="0 0 1000 640" width="100%" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Loop de governança: MMM, MTA, Testes A/B e Learning Repository" '
        'style="display:block;max-width:100%">',
        "<defs>",
        '<filter id="cardShadow" x="-30%" y="-30%" width="160%" height="180%">',
        '<feDropShadow dx="0" dy="6" stdDeviation="10" '
        f'flood-color="#000000" flood-opacity="{shadow_opacity}"/>',
        "</filter>",
        '<marker id="arrowHead" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{muted}"/>',
        "</marker>",
    ]

    # Gradiente sutil em cada card, na cor da sua camada
    for i, (_, _, _, color) in enumerate(LOOP_NODES):
        parts.append(
            f'<linearGradient id="grad{i}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="{0.30 if dark else 0.16}"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            "</linearGradient>"
        )
    parts.append("</defs>")

    # Trilha pontilhada do ciclo, ao fundo
    parts.append(
        f'<ellipse cx="500" cy="322" rx="298" ry="206" fill="none" stroke="{card_stroke}" '
        'stroke-width="1.5" stroke-dasharray="3 9" opacity="0.75"/>'
    )

    # Setas
    for (x1, y1, cx, cy, x2, y2, lx, ly, anchor), label in zip(edges, LOOP_EDGES):
        parts.append(
            f'<path d="M {x1} {y1} Q {cx} {cy} {x2} {y2}" fill="none" stroke="{muted}" '
            'stroke-width="2.2" stroke-linecap="round" opacity="0.85" '
            'marker-end="url(#arrowHead)"/>'
        )
        parts.append(
            f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" fill="{muted}" '
            'font-family="Segoe UI, Inter, sans-serif" font-size="14.5" '
            f'font-style="italic">{label}</text>'
        )

    # Cards
    for i, ((cx, cy), (title, subtitle, icon, color)) in enumerate(zip(cards, LOOP_NODES)):
        x, y = cx - w / 2, cy - h / 2
        parts.append(
            f'<g filter="url(#cardShadow)">'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
            f'fill="{card}" stroke="{card_stroke}" stroke-width="1.2"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
            f'fill="url(#grad{i})"/>'
            f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2.5" fill="{color}"/>'
            "</g>"
        )
        parts.append(
            f'<text x="{x + 26}" y="{cy + 2}" font-size="26" '
            'dominant-baseline="middle">{}</text>'.format(icon)
        )
        parts.append(
            f'<text x="{x + 62}" y="{cy - 8}" fill="{text}" '
            'font-family="Segoe UI, Inter, sans-serif" font-size="18.5" '
            f'font-weight="680">{title}</text>'
        )
        parts.append(
            f'<text x="{x + 62}" y="{cy + 16}" fill="{muted}" '
            'font-family="Segoe UI, Inter, sans-serif" font-size="13.5">'
            f'{subtitle}</text>'
        )

    # Núcleo do ciclo
    parts.append(
        f'<circle cx="500" cy="322" r="74" fill="{card}" stroke="{card_stroke}" '
        f'stroke-width="1.2" opacity="0.96"/>'
        f'<circle cx="500" cy="322" r="74" fill="{GOLD}" opacity="{glow}"/>'
    )
    parts.append(
        f'<text x="500" y="308" text-anchor="middle" fill="{text}" '
        'font-family="Segoe UI, Inter, sans-serif" font-size="17" font-weight="700">'
        "Loop de</text>"
        f'<text x="500" y="332" text-anchor="middle" fill="{text}" '
        'font-family="Segoe UI, Inter, sans-serif" font-size="17" font-weight="700">'
        "governança</text>"
        f'<text x="500" y="356" text-anchor="middle" fill="{muted}" '
        'font-family="Segoe UI, Inter, sans-serif" font-size="12.5">contínuo</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def ladder_svg(steps: list[tuple[str, str, str]], height: int = 190) -> str:
    """Escada Descritivo → Diagnóstico → Preditivo → Prescritivo.

    `steps` = [(número, título, pergunta), ...]
    """
    t = tokens()
    colors = [TEAL, NAVY, GOLD, "#2FA36B"]
    n = len(steps)
    width = 1000
    gap = 18
    box_w = (width - gap * (n - 1)) / n

    parts = [
        f'<svg viewBox="0 0 {width} 150" width="100%" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" style="display:block;max-width:100%">'
    ]
    for i, (number, title, question) in enumerate(steps):
        x = i * (box_w + gap)
        color = colors[i % len(colors)]
        lift = (n - i - 1) * 8  # degraus subindo da esquerda para a direita
        y = 30 + lift
        box_h = 100 - lift
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="14" '
            f'fill="{t.surface}" stroke="{t.border}" stroke-width="1.2"/>'
            f'<rect x="{x}" y="{y}" width="{box_w}" height="4" rx="2" fill="{color}"/>'
            f'<circle cx="{x + 26}" cy="{y + 34}" r="14" fill="{color}"/>'
            f'<text x="{x + 26}" y="{y + 39}" text-anchor="middle" fill="#FFFFFF" '
            f'font-family="Segoe UI, sans-serif" font-size="14" font-weight="800">{number}</text>'
            f'<text x="{x + 50}" y="{y + 39}" fill="{t.text}" '
            f'font-family="Segoe UI, sans-serif" font-size="16" font-weight="660">{title}</text>'
            f'<text x="{x + 18}" y="{y + 66}" fill="{t.text_muted}" '
            f'font-family="Segoe UI, sans-serif" font-size="13">{question}</text>'
        )
        if i < n - 1:
            ax = x + box_w + 3
            parts.append(
                f'<path d="M {ax} {y + 46} l 10 0" stroke="{t.text_muted}" stroke-width="2" '
                'stroke-linecap="round" opacity="0.8"/>'
            )
    parts.append("</svg>")
    return "".join(parts)
