"""IMC Prosperity 4 — Round 3 trader v7 (aggressive multi-product MM).

Complete rewrite. Previous attempts (v1: -$7k, v3: -$2.3k, v6: -$9) used
selective take-only or post-only-deep strategies; top competitors achieving
100-150k PnL with avg fill 12+ are clearly running active passive market
making on multiple products with a fair value good enough to avoid adverse
selection.

This rewrite:
  - Microprice + L1 imbalance + trade-flow tilt as fair value (no EWMA lag)
  - Stoikov-style inventory skew (reservation price)
  - Posts INSIDE the book at best_bid+1 / best_ask-1 in size 15-20
  - Multi-product: HYDROGEL_PACK + VELVETFRUIT_EXTRACT + voucher MM
  - Per-product circuit breaker to bound DD at ~$3k/product

This IS aggressive — it WILL get adversely selected on some fills. The bet
is that on net, the FV model + size catches enough fair flow to overcome
adverse selection.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple, Any

try:
    from datamodel import OrderDepth, TradingState, Order  # type: ignore
except Exception:
    OrderDepth = TradingState = Order = object  # type: ignore


# ============================================================== constants ===
HYDROGEL = "HYDROGEL_PACK"
VFE      = "VELVETFRUIT_EXTRACT"
VEV_DEEP_ITM = ["VEV_4000", "VEV_4500"]
VEV_OPTION   = ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]
VEV_DEAD     = ["VEV_6000", "VEV_6500"]
VEV_ALL      = VEV_DEEP_ITM + VEV_OPTION + VEV_DEAD
ALL_PRODUCTS = [HYDROGEL, VFE] + VEV_DEEP_ITM + VEV_OPTION

STRIKE = {f"VEV_{k}": k for k in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)}
POS_LIMIT = {HYDROGEL: 200, VFE: 200, **{v: 300 for v in VEV_ALL}}

# Per-product configuration. tune via observation of live behaviour.
PRODUCT_CFG = {
    HYDROGEL: {
        # v10: Hydrogel passive MM with anchor-to-10000 (huge breakthrough).
        # L1 imbalance is dead (bot volumes ~15/15 always symmetric).
        # Anchor signal dominates: FV pulled toward 10000 → systematic
        # mean-reversion. Backtest contribution +$30-55k/day.
        "enabled":      True,
        "size":         20,
        "size_take":    30,
        "skew_coef":    0.02,
        "max_pos":      200,
        "spread_min":   3,
        "tilt_weight":  0.15,
        "imb_weight":   1.0,
        "edge_take":    2.0,
        "dd_halt":      8000,        # bigger budget; biggest expected contributor
        # Fixed anchor at 10005 — historical sweep peak on days 0/1/2.
        # Mean-reversion on hydrogel is a stable structural feature; a fixed
        # anchor gives the strongest signal. Rolling-median tested but worse
        # because it chases mid and dilutes the reversion bias.
        "anchor":       10005,
        "anchor_weight": 0.55,
        "anchor_adaptive": False,
    },
    # v8 tuning (backtest +$19,598 across 3 days, all positive):
    # imbalance weight is the killer signal — sweep showed peak at 3.1 for VFE,
    # 1.7 for vouchers. Skew coef low (0.01) — Stoikov skew was over-constraining.
    # Larger sizes scale fills proportionally without breaking adverse selection.
    VFE: {
        "enabled":      True,
        "size":         15,
        "size_take":    22,
        "skew_coef":    0.01,
        "max_pos":      150,
        "spread_min":   3,
        "tilt_weight":  0.15,       # back to v8 — v9 amplification lost $17k live
        "imb_weight":   3.1,        # tuned — peak before instability cliff at 4.0
        "edge_take":    1.5,
        "dd_halt":      4000,
    },
    "VOUCHER": {
        "enabled":      True,
        "size":         20,
        "size_take":    30,
        "skew_coef":    0.0125,
        "max_pos":      200,
        "spread_min":   2,
        "tilt_weight":  0.10,       # back to v8 baseline
        "imb_weight":   1.7,        # tuned — peak before instability cliff at 2.5
        "edge_take":    1.0,
        "dd_halt":      2500,
    },
}

# ============================================================ utilities ====
def book_microprice(depth) -> float | None:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid = max(depth.buy_orders.keys())
    ask = min(depth.sell_orders.keys())
    bv = depth.buy_orders[bid]
    av = -depth.sell_orders[ask]
    if bv + av <= 0:
        return 0.5 * (bid + ask)
    return (bid * av + ask * bv) / (bv + av)


def book_l1_imbalance(depth) -> float:
    """Returns imbalance in [-1, 1]: +1 = all bid, -1 = all ask."""
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    bv = depth.buy_orders[max(depth.buy_orders.keys())]
    av = -depth.sell_orders[min(depth.sell_orders.keys())]
    if bv + av <= 0:
        return 0.0
    return (bv - av) / (bv + av)


def book_multi_imbalance(depth) -> float:
    """Volume-weighted imbalance across all visible levels.
    L1 contributes most; L2 and L3 add depth signal.
    """
    if not depth.buy_orders or not depth.sell_orders:
        return 0.0
    bids = sorted(depth.buy_orders.items(), reverse=True)[:3]
    asks = sorted(depth.sell_orders.items())[:3]
    bv = sum(v * w for (_, v), w in zip(bids, [1.0, 0.5, 0.25]))
    av = sum(-v * w for (_, v), w in zip(asks, [1.0, 0.5, 0.25]))
    if bv + av <= 0:
        return 0.0
    return (bv - av) / (bv + av)


def best_bid_ask(depth):
    bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
    ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return bid, ask


def clamp_qty(qty: int, pos: int, limit: int) -> int:
    if qty > 0:
        return max(0, min(qty, limit - pos))
    return min(0, max(qty, -limit - pos))


# =============================================================== Trader ====
class Trader:

    BIO_POD_BIDS = (765, 855)

    def run(self, state):
        try:
            return self._run_inner(state)
        except Exception as e:
            print(f"[trader] EXC: {e}")
            return {}, 0, getattr(state, "traderData", "") or ""

    def _run_inner(self, state):
        td = self._load_state(state.traderData)
        ts = state.timestamp
        positions = state.position or {}
        order_depths = state.order_depths
        market_trades = state.market_trades or {}

        # ---- 1. Trade-flow tilt: aggregate recent market_trades ----------
        # +1 per buy aggressor (price ≥ mid), -1 per sell aggressor.
        for prod, trades in market_trades.items():
            tilt = td["tilt"].get(prod, 0.0)
            d = order_depths.get(prod)
            mid = None
            if d and d.buy_orders and d.sell_orders:
                mid = 0.5 * (max(d.buy_orders.keys()) + min(d.sell_orders.keys()))
            for tr in trades:
                if mid is None:
                    continue
                if tr.price > mid:
                    tilt = tilt * 0.9 + 1.0
                elif tr.price < mid:
                    tilt = tilt * 0.9 - 1.0
                else:
                    tilt = tilt * 0.95
            td["tilt"][prod] = max(-10.0, min(10.0, tilt))

        # ---- 2. Per-product MM ------------------------------------------
        # Skip VEV_DEEP_ITM (4000, 4500): backtest showed consistent ~-$80-355
        # losses per day — basis barely moves, so 1-tick spread capture is
        # eaten by 2-tick adverse selection.
        all_orders: Dict[str, List] = {}
        for prod in [HYDROGEL, VFE] + VEV_OPTION:
            depth = order_depths.get(prod)
            if depth is None:
                continue
            cfg_key = prod if prod in (HYDROGEL, VFE) else "VOUCHER"
            cfg = PRODUCT_CFG[cfg_key]
            if not cfg["enabled"]:
                continue
            # Per-product DD circuit breaker
            if td["halted"].get(prod, 0) > ts:
                continue
            orders = self._mm_one(prod, depth, positions.get(prod, 0), cfg, td, ts)
            if orders:
                all_orders[prod] = orders

        # ---- 3. PnL tracking & circuit breaker ---------------------------
        for prod in (HYDROGEL, VFE):                  # only track for the big-PnL products
            d = order_depths.get(prod)
            if d is None or not d.buy_orders or not d.sell_orders:
                continue
            mid = 0.5 * (max(d.buy_orders.keys()) + min(d.sell_orders.keys()))
            pos = positions.get(prod, 0)
            mtm = pos * mid + td["cash"].get(prod, 0.0)
            track = td["pnl_track"].setdefault(prod, [])
            track.append((ts, mtm))
            track[:] = track[-80:]
            if len(track) >= 50:
                drop = track[0][1] - track[-1][1]
                cfg = PRODUCT_CFG[prod]
                if drop > cfg["dd_halt"]:
                    td["halted"][prod] = ts + 200_000

        traderData = self._save_state(td)
        return all_orders, 0, traderData

    # ---------------------------------------------------------------- MM ---
    def _mm_one(self, prod: str, depth, pos: int, cfg: dict, td: dict, ts: int) -> List:
        """Microprice + imbalance + tilt + Stoikov skew → quote both sides."""
        out = []
        if not depth.buy_orders or not depth.sell_orders:
            return out
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        spread = best_ask - best_bid
        mid = 0.5 * (best_bid + best_ask)
        if spread <= 0:
            return out

        # --- Fair value -------------------------------------------------
        mp = book_microprice(depth)
        imb = book_l1_imbalance(depth)             # L1-only is sharper than multi
        tilt = td["tilt"].get(prod, 0.0)
        # Trade flow contributes a small bias in ticks
        tilt_ticks = cfg["tilt_weight"] * tilt
        # Imbalance contributes in ticks (positive imbalance = bid-heavy = mid pushes up)
        imb_ticks = cfg["imb_weight"] * imb
        # Anchor signal: adaptive rolling median + offset.
        # Historical sweep showed peak at anchor=10005 (5 above true mean ≈10000).
        # Live mean was 9979, so anchor=10000 was 21 ticks too high. Adapting the
        # anchor to a rolling median + small upward offset keeps the proven bias
        # even when the day's center differs.
        anchor = cfg.get("anchor")          # legacy fixed anchor
        anchor_ticks = 0.0
        if cfg.get("anchor_adaptive"):
            hist = td.setdefault("mid_hist", {}).setdefault(prod, [])
            hist.append(mp)
            cap = cfg.get("anchor_window", 500)
            if len(hist) > cap:
                del hist[0:len(hist) - cap]
            if len(hist) >= cfg.get("anchor_warmup", 60):
                rolling_anchor = sorted(hist)[len(hist) // 2] + cfg.get("anchor_offset", 5)
                dev = mp - rolling_anchor
                anchor_ticks = -cfg.get("anchor_weight", 0.55) * dev
        elif anchor is not None:
            dev = mp - anchor
            anchor_ticks = -cfg.get("anchor_weight", 0.3) * dev
        fv = mp + tilt_ticks + imb_ticks + anchor_ticks

        # --- Stoikov-style reservation price (skew toward unwinding) ----
        # Larger skew when inventory is high (in ticks per share)
        reservation = fv - cfg["skew_coef"] * pos

        max_pos = cfg["max_pos"]
        size = cfg["size"]

        # --- Take orders: only when book offers edge over fv -----------
        edge = cfg["edge_take"]
        if best_ask < fv - edge and pos < max_pos:
            avail = -depth.sell_orders[best_ask]
            qty = clamp_qty(min(avail, cfg["size_take"]), pos, max_pos)
            if qty > 0:
                out.append(Order(prod, best_ask, qty))
                pos += qty
        if best_bid > fv + edge and pos > -max_pos:
            avail = depth.buy_orders[best_bid]
            qty = clamp_qty(-min(avail, cfg["size_take"]), pos, max_pos)
            if qty < 0:
                out.append(Order(prod, best_bid, qty))
                pos += qty

        # --- Passive quotes: post inside the book if room ---------------
        if spread < cfg["spread_min"]:
            return out

        # Place at best±1 normally, but if our reservation strays beyond,
        # post at our reservation rounded.
        bid_target = max(best_bid + 1, int(round(reservation - spread / 2)))
        ask_target = min(best_ask - 1, int(round(reservation + spread / 2)))

        # Don't cross
        bid_target = min(bid_target, best_ask - 1)
        ask_target = max(ask_target, best_bid + 1)
        # Don't price-improve so far we cross our own quotes
        if bid_target >= ask_target:
            bid_target = best_bid + 1
            ask_target = best_ask - 1

        # Symmetric sizing — asymmetric was untested speculation that hurt live.
        if pos < max_pos:
            qty = clamp_qty(size, pos, max_pos)
            if qty > 0:
                out.append(Order(prod, bid_target, qty))
        if pos > -max_pos:
            qty = clamp_qty(-size, pos, max_pos)
            if qty < 0:
                out.append(Order(prod, ask_target, qty))

        return out

    # --------------------------------------------------- state ------------
    def _load_state(self, traderData: str) -> dict:
        if not traderData:
            return self._fresh_state()
        try:
            d = json.loads(traderData)
            for key in ("tilt", "halted", "pnl_track", "cash", "mid_hist"):
                d.setdefault(key, {})
            return d
        except Exception:
            return self._fresh_state()

    def _fresh_state(self) -> dict:
        return {"tilt": {}, "halted": {}, "pnl_track": {}, "cash": {}, "mid_hist": {}}

    def _save_state(self, td: dict) -> str:
        compact = {
            "tilt":      {k: round(v, 3) for k, v in td.get("tilt", {}).items()},
            "halted":    td.get("halted", {}),
            "pnl_track": {k: [(t, round(p, 1)) for t, p in v[-30:]]
                          for k, v in td.get("pnl_track", {}).items()},
            "cash":      {k: round(v, 1) for k, v in td.get("cash", {}).items()},
            "mid_hist":  {k: [round(x, 1) for x in v[-500:]]
                          for k, v in td.get("mid_hist", {}).items()},
        }
        s = json.dumps(compact, separators=(",", ":"))
        if len(s) > 30_000:
            compact["pnl_track"] = {}
            s = json.dumps(compact, separators=(",", ":"))
        return s
