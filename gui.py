"""Pygame visual UI for CityMind.

Reuses the same pipeline as main.py without modifying it.
Run: python gui.py [--seed N] [--steps N] [--nodes N]
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
import sys
from typing import Callable

import pygame

from main import (
    SIMULATION_MODE_COMPLETE,
    SIMULATION_MODE_STRICT,
    VALID_SIMULATION_MODES,
    RunConfig,
)
from ui import theme
from ui.controller import SimulationController, UIEvent
from ui.renderer import (
    CanvasRect,
    GridRenderer,
    LegendRenderer,
    OVERLAY_COVERAGE,
    OVERLAY_LAYOUT,
    OVERLAY_NAMES,
    OVERLAY_RISK,
    OVERLAY_ROADS,
)


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
LOG_PANEL_WIDTH = 340
HUD_HEIGHT = 108
TOOLBAR_HEIGHT = 44
LEGEND_WIDTH = 180
FPS = 60

AUTO_RUN_SPEEDS = [0.5, 1.0, 2.0, 4.0, 8.0]

# ── Refined colour palette ──────────────────────────────────────────────────
DEEP_BG        = (11, 13, 20)          # near-black canvas
PANEL_BG       = (18, 22, 34)          # panel fill
PANEL_BG_ALT   = (22, 27, 42)          # slightly lighter variant
PANEL_BORDER   = (38, 48, 72)          # subtle border
PANEL_GLOW     = (52, 68, 110)         # focused / active border
ACCENT         = (251, 196, 54)        # amber accent
ACCENT_DIM     = (180, 138, 38)        # dimmed amber
TEXT_PRIMARY   = (225, 230, 242)       # primary text
TEXT_SECONDARY = (138, 148, 172)       # secondary / muted
TEXT_DARK      = (11, 13, 20)          # text on bright bg

LOG_ENTRY_COLORS = {
    "ROAD_BLOCKED":  (222, 100, 100),
    "C4_UNREACHABLE":(222, 100, 100),
    "C4_REACHED":    (100, 210, 130),
    "C3_RECOMPUTE":  (230, 205, 100),
    "C4_REPLAN":     (230, 205, 100),
    "TICK":          (80, 90, 115),
}
LOG_DEFAULT_COLOR = TEXT_SECONDARY

# ── Thin separator line ─────────────────────────────────────────────────────
def _draw_hsep(surface: pygame.Surface, x: int, y: int, w: int,
               color: tuple = PANEL_BORDER) -> None:
    pygame.draw.line(surface, color, (x, y), (x + w, y))


def _draw_vsep(surface: pygame.Surface, x: int, y: int, h: int,
               color: tuple = PANEL_BORDER) -> None:
    pygame.draw.line(surface, color, (x, y), (x, y + h))


# ── Panel drawing helper ─────────────────────────────────────────────────────
def _draw_panel(surface: pygame.Surface, rect: pygame.Rect, *,
                bg: tuple = PANEL_BG, border: tuple = PANEL_BORDER,
                radius: int = 8, border_width: int = 1) -> None:
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, border_width, border_radius=radius)


# ── Pill / badge helper ──────────────────────────────────────────────────────
def _draw_pill(surface: pygame.Surface, rect: pygame.Rect,
               fill: tuple, border: tuple | None = None) -> None:
    radius = rect.height // 2
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border, rect, 1, border_radius=radius)


# ── Tag / chip ───────────────────────────────────────────────────────────────
def _draw_tag(surface: pygame.Surface, font: pygame.font.Font,
              text: str, x: int, y: int,
              fill: tuple = PANEL_BG_ALT,
              text_color: tuple = TEXT_SECONDARY,
              border: tuple = PANEL_BORDER) -> int:
    """Draw a small label chip, return its right edge x."""
    tw, th = font.size(text)
    pad_x, pad_y = 8, 3
    r = pygame.Rect(x, y, tw + pad_x * 2, th + pad_y * 2)
    _draw_panel(surface, r, bg=fill, border=border, radius=r.height // 2)
    surface.blit(font.render(text, True, text_color), (x + pad_x, y + pad_y))
    return r.right


def _fit_text(font: pygame.font.Font, text: str, max_w: int) -> str:
    if max_w <= 0 or not text:
        return text
    if font.size(text)[0] <= max_w:
        return text
    ellipsis = "…"
    lo, hi, best = 0, len(text), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if font.size(text[:mid] + ellipsis)[0] <= max_w:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return (text[:best] + ellipsis) if best > 0 else ellipsis


def parse_cli() -> RunConfig:
    parser = argparse.ArgumentParser(description="CityMind Pygame UI")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=None)
    parser.add_argument("--mode", choices=VALID_SIMULATION_MODES, default=None)
    parser.add_argument("--no-risk-refresh", action="store_true")
    args = parser.parse_args()
    config = RunConfig()
    if args.seed is not None:         config = replace(config, seed=args.seed)
    if args.steps is not None:        config = replace(config, simulation_steps=args.steps)
    if args.nodes is not None:        config = replace(config, node_count=args.nodes)
    if args.mode is not None:         config = replace(config, simulation_mode=args.mode)
    if args.no_risk_refresh:          config = replace(config, risk_refresh_every_step=False)
    return config


# ═══════════════════════════════════════════════════════════════════════════
class EventLogPanel:
    MAX_ENTRIES = 500
    ENTRY_H = 17

    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.scroll = 0

    def handle_scroll(self, delta: int) -> None:
        self.scroll = max(0, self.scroll + delta)

    def draw(self, surface: pygame.Surface, events: list[UIEvent],
             font: pygame.font.Font, title_font: pygame.font.Font) -> None:
        # Outer panel
        _draw_panel(surface, self.rect, bg=PANEL_BG, border=PANEL_BORDER, radius=10)

        # Header strip
        header = pygame.Rect(self.rect.x, self.rect.y,
                             self.rect.width, 38)
        pygame.draw.rect(surface, PANEL_BG_ALT, header,
                         border_top_left_radius=10, border_top_right_radius=10)
        _draw_hsep(surface, self.rect.x + 1, self.rect.y + 38,
                   self.rect.width - 2, PANEL_BORDER)

        # Amber accent bar on left edge of header
        bar = pygame.Rect(self.rect.x, self.rect.y, 3, 38)
        pygame.draw.rect(surface, ACCENT, bar,
                         border_top_left_radius=10)

        lbl = title_font.render("EVENT LOG", True, ACCENT)
        surface.blit(lbl, (self.rect.x + 14, self.rect.y + 11))

        # Entry area
        ea = pygame.Rect(self.rect.x + 6, self.rect.y + 44,
                         self.rect.width - 12, self.rect.height - 52)
        pygame.draw.rect(surface, DEEP_BG, ea, border_radius=6)

        trimmed = events[-self.MAX_ENTRIES:]
        visible = max(ea.height // self.ENTRY_H, 1)
        max_scroll = max(0, len(trimmed) - visible)
        self.scroll = min(self.scroll, max_scroll)

        start = max(0, len(trimmed) - visible - self.scroll)
        end   = min(len(trimmed), start + visible)
        shown = list(reversed(trimmed[start:end]))

        max_tw = ea.width - 14
        ix, iy = ea.x + 7, ea.y + 4

        prev_clip = surface.get_clip()
        surface.set_clip(ea)

        for i, ev in enumerate(shown):
            row_y = iy + i * self.ENTRY_H
            # Alternating row tint
            if i % 2 == 0:
                row_rect = pygame.Rect(ea.x + 2, row_y - 1,
                                       ea.width - 4, self.ENTRY_H)
                tint = pygame.Surface((row_rect.width, row_rect.height), pygame.SRCALPHA)
                tint.fill((255, 255, 255, 6))
                surface.blit(tint, row_rect.topleft)

            step_txt = f"[{ev.step:>3}]"
            kind_txt = f" {ev.kind:<13} "
            msg_txt  = ev.message

            color = LOG_ENTRY_COLORS.get(ev.kind, LOG_DEFAULT_COLOR)

            step_surf = font.render(step_txt, True, TEXT_SECONDARY)
            kind_surf = font.render(kind_txt, True, color)

            step_w = step_surf.get_width()
            kind_w = kind_surf.get_width()
            msg_max = max_tw - step_w - kind_w
            msg_txt  = _fit_text(font, msg_txt, msg_max)
            msg_surf  = font.render(msg_txt, True, TEXT_PRIMARY)

            surface.blit(step_surf, (ix, row_y))
            surface.blit(kind_surf, (ix + step_w, row_y))
            surface.blit(msg_surf,  (ix + step_w + kind_w, row_y))

        surface.set_clip(prev_clip)

        # Scroll caret
        if self.scroll > 0:
            c = font.render("▲", True, ACCENT_DIM)
            surface.blit(c, (ea.right - 14, ea.y + 3))
        if start > 0:
            m = font.render(f"+{start} older", True, TEXT_SECONDARY)
            surface.blit(m, (ix, ea.bottom - self.ENTRY_H - 2))

        # Thin right-edge scrollbar track
        if len(trimmed) > visible:
            track_h = ea.height - 8
            track_x = ea.right - 5
            pygame.draw.line(surface, PANEL_BORDER,
                             (track_x, ea.y + 4),
                             (track_x, ea.y + 4 + track_h))
            ratio = visible / max(len(trimmed), 1)
            thumb_h = max(16, int(track_h * ratio))
            scroll_frac = (max_scroll - self.scroll) / max(max_scroll, 1)
            thumb_y = ea.y + 4 + int((track_h - thumb_h) * scroll_frac)
            pygame.draw.line(surface, ACCENT_DIM,
                             (track_x, thumb_y),
                             (track_x, thumb_y + thumb_h), 2)


# ═══════════════════════════════════════════════════════════════════════════
class HUDRenderer:
    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect

    def draw(self, surface: pygame.Surface, controller: SimulationController,
             overlay_mode: int, auto_run: bool, run_all: bool, speed: float,
             font: pygame.font.Font, title_font: pygame.font.Font,
             big_font: pygame.font.Font) -> None:
        _draw_panel(surface, self.rect, bg=PANEL_BG, border=PANEL_BORDER, radius=10)

        # Left accent bar
        bar = pygame.Rect(self.rect.x, self.rect.y, 3, self.rect.height)
        pygame.draw.rect(surface, ACCENT, bar,
                         border_top_left_radius=10,
                         border_bottom_left_radius=10)

        # Brand / title
        brand = big_font.render("CITY", True, ACCENT)
        brand2 = big_font.render("MIND", True, TEXT_PRIMARY)
        surface.blit(brand,  (self.rect.x + 14, self.rect.y + 10))
        surface.blit(brand2, (self.rect.x + 14 + brand.get_width(), self.rect.y + 10))

        sub = font.render("Urban Intelligence System", True, TEXT_SECONDARY)
        surface.blit(sub, (self.rect.x + 14, self.rect.y + 38))

        # Vertical separator
        _draw_vsep(surface, self.rect.x + 210, self.rect.y + 10,
                   self.rect.height - 20, PANEL_BORDER)

        # ── Stats block ──────────────────────────────────────────────────
        router = controller.router
        c3 = controller.stage_results.get("c3") or {}
        fitness = c3.get("fitness") if isinstance(c3, dict) else None
        c5 = controller.stage_results.get("c5") \
            if isinstance(controller.stage_results.get("c5"), dict) else {}

        pending     = len(router.pending_civilians)   if router else 0
        reached     = len(router.reached_civilians)   if router else 0
        unreachable = len(router.unreachable_civilians) if router else 0
        alloc = controller.police_officer_allocation()
        officer_total = sum(int(v) for v in alloc.values()) if alloc else 0
        c5_fallback = bool(c5.get("fallback_used")) if isinstance(c5, dict) else False

        cfg = controller.config
        step_str = (
            f"{controller.current_step} / {cfg.simulation_steps}  (cap {cfg.completion_step_cap})"
            if cfg.simulation_mode == SIMULATION_MODE_COMPLETE
            else f"{controller.current_step} / {cfg.simulation_steps}"
        )

        sx = self.rect.x + 220
        sy = self.rect.y + 10

        # Stat entries: (label, value, value_color)
        stats = [
            ("STEP",    step_str,                              TEXT_PRIMARY),
            ("SEED",    str(cfg.seed),                         TEXT_SECONDARY),
            ("GRID",    f"{cfg.grid_rows}×{cfg.grid_cols}",    TEXT_SECONDARY),
            ("MODE",    cfg.simulation_mode,                    ACCENT),
        ]
        col_w = 140
        for i, (lbl, val, vc) in enumerate(stats):
            cx = sx + (i % 4) * col_w
            cy = sy + (i // 4) * 28
            surface.blit(font.render(lbl, True, TEXT_SECONDARY), (cx, cy))
            surface.blit(font.render(val, True, vc),              (cx, cy + 15))

        # Second row
        stats2 = [
            ("PENDING",     str(pending),                              (230, 180, 90)),
            ("REACHED",     str(reached),                              (100, 210, 130)),
            ("UNREACHABLE", str(unreachable),                          (222, 100, 100)),
            ("WORST RESP",  f"{fitness:.2f}" if fitness is not None else "—", TEXT_PRIMARY),
        ]
        sy2 = sy + 52
        for i, (lbl, val, vc) in enumerate(stats2):
            cx = sx + i * col_w
            surface.blit(font.render(lbl, True, TEXT_SECONDARY), (cx, sy2))
            surface.blit(font.render(val, True, vc),              (cx, sy2 + 15))

        # Right-side status chips
        chip_x = self.rect.right - 180
        chip_y = self.rect.y + 10

        def chip(text: str, active: bool, cy: int) -> None:
            fill   = ACCENT       if active else (30, 36, 52)
            tcolor = TEXT_DARK    if active else TEXT_SECONDARY
            bcolor = ACCENT       if active else PANEL_BORDER
            r = pygame.Rect(chip_x, cy, 160, 22)
            _draw_pill(surface, r, fill, bcolor)
            s = font.render(text, True, tcolor)
            surface.blit(s, (r.centerx - s.get_width() // 2,
                              r.centery - s.get_height() // 2))

        chip(f"AUTO-RUN  ×{speed:g}", auto_run, chip_y)
        chip("RUN-ALL",               run_all,  chip_y + 28)

        risk_on = cfg.risk_refresh_every_step
        chip(f"RISK-REFRESH {'ON' if risk_on else 'OFF'}", risk_on, chip_y + 56)

        police_s = font.render(f"POLICE  {officer_total} officers", True,
                                TEXT_SECONDARY if not c5_fallback else (222, 100, 100))
        surface.blit(police_s, (chip_x, chip_y + 86))


# ═══════════════════════════════════════════════════════════════════════════
class CustomizationDialog:
    def __init__(self) -> None:
        self.visible = False
        self.selected = 0
        self.fields: list[tuple[str, int, int, int]] = []
        self.values: dict[str, int] = {}

    def open(self, config: RunConfig) -> None:
        self.visible = True
        self.selected = 0
        self.fields = [
            ("seed",                   0, 9999, 1),
            ("grid_rows",              2, 10,   1),
            ("grid_cols",              2, 10,   1),
            ("simulation_steps",       1, 100,  1),
            ("speed_index",            0, len(AUTO_RUN_SPEEDS) - 1, 1),
            ("flood_every_n_steps",    0, 10,   1),
            ("c3_recompute_interval",  0, 10,   1),
            ("ambulance_count",        1, 5,    1),
            ("simulation_mode_index",  0, len(VALID_SIMULATION_MODES) - 1, 1),
            ("risk_refresh_every_step",0, 1,    1),
            ("completion_step_cap",    20, 2000, 10),
        ]
        try:    mode_index = VALID_SIMULATION_MODES.index(config.simulation_mode)
        except ValueError: mode_index = 0
        self.values = {
            "seed":                    config.seed,
            "grid_rows":               config.grid_rows,
            "grid_cols":               config.grid_cols,
            "simulation_steps":        config.simulation_steps,
            "speed_index":             1,
            "flood_every_n_steps":     config.flood_every_n_steps,
            "c3_recompute_interval":   config.c3_recompute_interval,
            "ambulance_count":         config.ambulance_count,
            "simulation_mode_index":   mode_index,
            "risk_refresh_every_step": 1 if config.risk_refresh_every_step else 0,
            "completion_step_cap":     int(config.completion_step_cap),
        }

    def close(self) -> None:
        self.visible = False

    def handle_key(self, event: pygame.event.Event,
                   apply_callback: Callable[[dict[str, int]], None]) -> bool:
        if not self.visible:
            return False
        if event.key == pygame.K_ESCAPE:
            self.close(); return True
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(self.fields); return True
        if event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(self.fields); return True
        if event.key in (pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d):
            name, low, high, stride = self.fields[self.selected]
            direction = -1 if event.key in (pygame.K_LEFT, pygame.K_a) else 1
            self.values[name] = max(low, min(high, self.values[name] + direction * stride))
            return True
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            apply_callback(dict(self.values)); self.close(); return True
        return False

    def draw(self, surface: pygame.Surface,
             font: pygame.font.Font, title_font: pygame.font.Font) -> None:
        if not self.visible:
            return

        # Dimmed overlay
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 8, 200))
        surface.blit(overlay, (0, 0))

        w, h = 560, 480
        x = (surface.get_width()  - w) // 2
        y = (surface.get_height() - h) // 2
        rect = pygame.Rect(x, y, w, h)

        _draw_panel(surface, rect, bg=(16, 20, 32), border=PANEL_GLOW, radius=12, border_width=2)

        # Header
        hdr = pygame.Rect(x, y, w, 50)
        pygame.draw.rect(surface, PANEL_BG_ALT, hdr,
                         border_top_left_radius=12, border_top_right_radius=12)
        _draw_hsep(surface, x + 1, y + 50, w - 2, PANEL_BORDER)

        # Accent bar
        pygame.draw.rect(surface, ACCENT, pygame.Rect(x, y, 4, 50),
                         border_top_left_radius=12)

        title = title_font.render("CUSTOMIZATION", True, ACCENT)
        surface.blit(title, (x + 18, y + 16))

        hint = font.render(
            "↑↓ select   ←→ adjust   Enter apply   Esc cancel",
            True, TEXT_SECONDARY)
        surface.blit(hint, (x + w - hint.get_width() - 16, y + 18))

        row_h = 34
        for idx, (name, low, high, _) in enumerate(self.fields):
            val   = self.values[name]
            sel   = idx == self.selected
            ry    = y + 58 + idx * row_h

            if sel:
                sel_rect = pygame.Rect(x + 8, ry + 2, w - 16, row_h - 4)
                pygame.draw.rect(surface, (28, 36, 58), sel_rect, border_radius=6)
                pygame.draw.rect(surface, ACCENT, sel_rect, 1, border_radius=6)

            label_color = ACCENT   if sel else TEXT_SECONDARY
            val_color   = TEXT_PRIMARY if sel else TEXT_SECONDARY

            # Display value
            display: object = val
            if name == "speed_index":
                display = f"{val}  (×{AUTO_RUN_SPEEDS[val]:g})"
            elif name == "simulation_mode_index":
                display = VALID_SIMULATION_MODES[val]
            elif name == "risk_refresh_every_step":
                display = "ON" if val else "OFF"

            prefix = "▸" if sel else " "
            lbl_surf = font.render(f"{prefix}  {name}", True, label_color)
            rng_surf = font.render(f"[{low}..{high}]", True, (60, 72, 100))
            val_surf = font.render(str(display), True, val_color)

            surface.blit(lbl_surf, (x + 20, ry + 8))
            surface.blit(rng_surf, (x + w - 110, ry + 8))
            surface.blit(val_surf, (x + w - 110 - val_surf.get_width() - 16, ry + 8))


# ═══════════════════════════════════════════════════════════════════════════
class Toolbar:
    def __init__(self, rect: pygame.Rect) -> None:
        self.rect = rect
        self.buttons: list[tuple[pygame.Rect, str, int | None]] = []

    def draw(self, surface: pygame.Surface, font: pygame.font.Font,
             overlay_mode: int, auto_run: bool) -> None:
        _draw_panel(surface, self.rect, bg=PANEL_BG, border=PANEL_BORDER, radius=8)
        self.buttons = []

        labels = [
            ("1  LAYOUT",   OVERLAY_LAYOUT),
            ("2  ROADS",    OVERLAY_ROADS),
            ("3  COVERAGE", OVERLAY_COVERAGE),
            ("4  RISK",     OVERLAY_RISK),
        ]
        x   = self.rect.x + 10
        cy  = self.rect.centery
        btn_h = self.rect.height - 12

        for text, mode in labels:
            active = overlay_mode == mode
            tw, th = font.size(text)
            bw     = tw + 22
            br     = pygame.Rect(x, cy - btn_h // 2, bw, btn_h)

            if active:
                # Active: amber pill
                pygame.draw.rect(surface, ACCENT, br, border_radius=6)
                pygame.draw.rect(surface, ACCENT_DIM, br, 1, border_radius=6)
                t = font.render(text, True, TEXT_DARK)
            else:
                pygame.draw.rect(surface, (26, 32, 50), br, border_radius=6)
                pygame.draw.rect(surface, PANEL_BORDER, br, 1, border_radius=6)
                t = font.render(text, True, TEXT_SECONDARY)

            surface.blit(t, (br.x + 11, br.centery - t.get_height() // 2))
            self.buttons.append((br, text, mode))
            x += bw + 6

        # Divider
        _draw_vsep(surface, x + 8, self.rect.y + 8, self.rect.height - 16, PANEL_BORDER)

        # Controls hint — right-aligned to panel
        hint_parts = [
            ("Space", "step"),
            ("R", "auto"),
            ("A", "run-all"),
            ("M", "mode"),
            ("X", "risk"),
            ("+/-", "speed"),
            ("C", "config"),
            ("Esc", "quit"),
        ]
        hx = x + 20
        hy = cy - font.get_height() // 2
        for key, action in hint_parts:
            ks = font.render(key, True, ACCENT_DIM)
            as_ = font.render(f" {action}  ", True, TEXT_SECONDARY)
            surface.blit(ks,  (hx, hy))
            hx += ks.get_width()
            surface.blit(as_, (hx, hy))
            hx += as_.get_width()

    def click(self, pos: tuple[int, int]) -> int | None:
        for br, _, mode in self.buttons:
            if br.collidepoint(pos) and mode is not None:
                return mode
        return None


# ═══════════════════════════════════════════════════════════════════════════
class CityMindGUI:
    def __init__(self, config: RunConfig) -> None:
        pygame.init()
        pygame.display.set_caption("CityMind — Urban Intelligence")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock  = pygame.time.Clock()

        # Font stack — try nicer monospace fonts, fall back gracefully
        mono_pref = ["Consolas", "Courier New", "Courier"]
        self.font       = self._best_font(mono_pref, 13)
        self.title_font = self._best_font(mono_pref, 15, bold=True)
        self.small_font = self._best_font(mono_pref, 11)
        self.big_font   = self._best_font(mono_pref, 20, bold=True)

        self.config       = config
        self.controller   = SimulationController(config)
        self.overlay_mode = OVERLAY_LAYOUT
        self.auto_run     = False
        self.run_all      = False
        self.speed_index  = 1
        self._accumulator = 0.0

        PAD = 12
        GRID_LEFT = PAD
        LOG_X     = WINDOW_WIDTH - LOG_PANEL_WIDTH - PAD
        GRID_W    = LOG_X - GRID_LEFT - PAD

        self.hud     = HUDRenderer(pygame.Rect(GRID_LEFT, PAD, GRID_W, HUD_HEIGHT))
        self.toolbar = Toolbar(pygame.Rect(
            GRID_LEFT, PAD + HUD_HEIGHT + 6, GRID_W, TOOLBAR_HEIGHT))

        canvas_y = PAD + HUD_HEIGHT + 6 + TOOLBAR_HEIGHT + 8
        canvas_h = WINDOW_HEIGHT - canvas_y - PAD
        self.renderer = GridRenderer(CanvasRect(GRID_LEFT, canvas_y, GRID_W, canvas_h))
        self.legend   = LegendRenderer()

        self.log_panel = EventLogPanel(
            pygame.Rect(LOG_X, PAD, LOG_PANEL_WIDTH, WINDOW_HEIGHT - PAD * 2))
        self.dialog = CustomizationDialog()

    @staticmethod
    def _best_font(names: list[str], size: int, bold: bool = False) -> pygame.font.Font:
        for name in names:
            try:
                f = pygame.font.SysFont(name, size, bold=bold)
                if f:
                    return f
            except Exception:
                pass
        return pygame.font.SysFont(None, size, bold=bold)

    # ── Main loop ────────────────────────────────────────────────────────
    def run(self) -> None:
        running = True
        pulse   = 0.0
        while running:
            dt    = self.clock.tick(FPS) / 1000.0
            pulse += dt * 4.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if self.dialog.handle_key(event, self._apply_customization):
                        continue
                    running = self._handle_key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.dialog.visible:
                    if event.button == 1:
                        mode = self.toolbar.click(event.pos)
                        if mode is not None:
                            self.overlay_mode = mode
                    elif event.button == 4:
                        self.log_panel.handle_scroll(3)
                    elif event.button == 5:
                        self.log_panel.handle_scroll(-3)

            if self.auto_run and not self.dialog.visible:
                self._accumulator += dt * AUTO_RUN_SPEEDS[self.speed_index]
                while self._accumulator >= 1.0:
                    advanced = self.controller.step()
                    self._accumulator -= 1.0
                    if not advanced:
                        self.auto_run = False
                        break
            elif self.run_all and not self.dialog.visible:
                burst = max(1, int(AUTO_RUN_SPEEDS[self.speed_index] * 3))
                for _ in range(burst):
                    advanced = self.controller.step(ignore_step_limit=True)
                    if not advanced:
                        self.run_all = False
                        break

            self._draw(pulse)
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, event: pygame.event.Event) -> bool:
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_SPACE:
            self.controller.step()
        elif event.key == pygame.K_r:
            self.auto_run = not self.auto_run
            if self.auto_run: self.run_all = False
            self._accumulator = 0.0
        elif event.key == pygame.K_a:
            self.run_all = not self.run_all
            if self.run_all: self.auto_run = False
            self._accumulator = 0.0
        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed_index = min(self.speed_index + 1, len(AUTO_RUN_SPEEDS) - 1)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_index = max(self.speed_index - 1, 0)
        elif event.key == pygame.K_1:   self.overlay_mode = OVERLAY_LAYOUT
        elif event.key == pygame.K_2:   self.overlay_mode = OVERLAY_ROADS
        elif event.key == pygame.K_3:   self.overlay_mode = OVERLAY_COVERAGE
        elif event.key == pygame.K_4:   self.overlay_mode = OVERLAY_RISK
        elif event.key == pygame.K_c:
            self.dialog.open(self.config)
            self.dialog.values["speed_index"] = self.speed_index
        elif event.key == pygame.K_m:
            new_mode = (SIMULATION_MODE_COMPLETE
                        if self.config.simulation_mode == SIMULATION_MODE_STRICT
                        else SIMULATION_MODE_STRICT)
            self.config = replace(self.config, simulation_mode=new_mode)
            self.controller.config = self.config
        elif event.key == pygame.K_x:
            self.config = replace(
                self.config,
                risk_refresh_every_step=not self.config.risk_refresh_every_step)
            self.controller.config = self.config
        elif event.key == pygame.K_PAGEUP:
            self.log_panel.handle_scroll(5)
        elif event.key == pygame.K_PAGEDOWN:
            self.log_panel.handle_scroll(-5)
        return True

    def _apply_customization(self, values: dict[str, int]) -> None:
        rows        = int(values.get("grid_rows",        self.config.grid_rows))
        cols        = int(values.get("grid_cols",        self.config.grid_cols))
        speed_index = int(values.get("speed_index",      self.speed_index))
        speed_index = max(0, min(speed_index, len(AUTO_RUN_SPEEDS) - 1))
        mode_index  = int(values.get("simulation_mode_index", 0))
        mode_index  = max(0, min(mode_index, len(VALID_SIMULATION_MODES) - 1))
        new_config  = replace(
            self.config,
            seed=int(values["seed"]),
            grid_rows=rows,
            grid_cols=cols,
            node_count=rows * cols,
            simulation_steps=int(values["simulation_steps"]),
            flood_every_n_steps=int(values["flood_every_n_steps"]),
            c3_recompute_interval=int(values["c3_recompute_interval"]),
            ambulance_count=int(values["ambulance_count"]),
            simulation_mode=VALID_SIMULATION_MODES[mode_index],
            risk_refresh_every_step=bool(values.get("risk_refresh_every_step", 1)),
            completion_step_cap=int(values.get("completion_step_cap",
                                               self.config.completion_step_cap)),
        )
        self.config       = new_config
        self.controller.reset(new_config)
        self.speed_index  = speed_index
        self.auto_run     = False
        self.run_all      = False
        self._accumulator = 0.0

    # ── Render ───────────────────────────────────────────────────────────
    def _draw(self, pulse: float) -> None:
        self.screen.fill(DEEP_BG)

        # Subtle grid-dot texture on background
        self._draw_bg_texture()

        self.hud.draw(
            self.screen, self.controller,
            self.overlay_mode, self.auto_run, self.run_all,
            AUTO_RUN_SPEEDS[self.speed_index],
            self.font, self.title_font, self.big_font,
        )
        self.toolbar.draw(self.screen, self.font, self.overlay_mode, self.auto_run)

        # Canvas border panel behind the grid
        canvas = self.renderer.canvas
        canvas_rect = pygame.Rect(canvas.x - 4, canvas.y - 4,
                                  canvas.width + 8, canvas.height + 8)
        _draw_panel(surface=self.screen, rect=canvas_rect,
                    bg=PANEL_BG, border=PANEL_BORDER, radius=8)

        self.renderer.draw(
            self.screen, self.controller,
            self.overlay_mode, self.font, pulse,
        )
        self.legend.draw(
            self.screen, self.small_font,
            canvas.x + 10,
            canvas.y + canvas.height - 162,
            self.overlay_mode,
        )
        self.log_panel.draw(
            self.screen, self.controller.events,
            self.small_font, self.title_font,
        )
        self.dialog.draw(self.screen, self.font, self.title_font)

        # Step progress bar at very bottom of canvas panel
        self._draw_step_bar(canvas_rect)

    def _draw_bg_texture(self) -> None:
        """Dot-grid texture for depth."""
        dot_color = (22, 27, 40)
        spacing = 28
        for gx in range(0, WINDOW_WIDTH, spacing):
            for gy in range(0, WINDOW_HEIGHT, spacing):
                pygame.draw.circle(self.screen, dot_color, (gx, gy), 1)

    def _draw_step_bar(self, canvas_rect: pygame.Rect) -> None:
        """Thin amber progress bar at the bottom of the canvas panel."""
        cfg        = self.controller.config
        step       = self.controller.current_step
        total      = max(cfg.simulation_steps, 1)
        frac       = min(step / total, 1.0)
        bar_rect   = pygame.Rect(canvas_rect.x + 1,
                                  canvas_rect.bottom - 6,
                                  canvas_rect.width - 2, 5)
        fill_rect  = pygame.Rect(bar_rect.x, bar_rect.y,
                                  int(bar_rect.width * frac), bar_rect.height)
        pygame.draw.rect(self.screen, (28, 34, 52), bar_rect,
                         border_bottom_left_radius=8, border_bottom_right_radius=8)
        if frac > 0:
            pygame.draw.rect(self.screen, ACCENT, fill_rect,
                             border_bottom_left_radius=8,
                             border_bottom_right_radius=8 if frac >= 0.99 else 0)


def main() -> None:
    config = parse_cli()
    gui    = CityMindGUI(config)
    gui.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)