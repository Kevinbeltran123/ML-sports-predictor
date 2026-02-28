"""Polymarket position tracker: SQLite persistence para posiciones y P&L.

DB: data/nba/predictions/PolymarketTracking.sqlite

Tablas:
  - positions: posiciones abiertas y cerradas
  - orders: log de todas las ordenes (incluye dry runs)
  - bankroll_history: snapshots diarios de bankroll
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import PREDICTIONS_DIR

logger = logging.getLogger(__name__)

POLYMARKET_DB = PREDICTIONS_DIR / "PolymarketTracking.sqlite"


class PolymarketTracker:
    """Persiste posiciones, ordenes y bankroll de Polymarket.

    Uso:
        tracker = PolymarketTracker()
        tracker.record_order(...)
        tracker.open_position(...)
        tracker.close_position(...)
        tracker.print_portfolio_report()
    """

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or POLYMARKET_DB
        self._init_db()

    def _init_db(self):
        """Crea tablas si no existen."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_date TEXT NOT NULL,
                    game_key TEXT NOT NULL,          -- "Home:Away"
                    team TEXT NOT NULL,               -- equipo al que apostamos
                    token_id TEXT,
                    entry_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    cost_basis REAL NOT NULL,
                    model_prob REAL,
                    current_price REAL,
                    unrealized_pnl REAL DEFAULT 0.0,
                    exit_price REAL,
                    realized_pnl REAL,
                    exit_reason TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN',   -- OPEN / CLOSED
                    dry_run INTEGER NOT NULL DEFAULT 1,    -- 1=dry run, 0=real
                    created_at TEXT NOT NULL,
                    closed_at TEXT,
                    UNIQUE(game_date, game_key, team, dry_run)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    game_date TEXT,
                    game_key TEXT,
                    team TEXT,
                    token_id TEXT,
                    side TEXT NOT NULL,              -- BUY / SELL
                    price REAL,
                    shares INTEGER,
                    cost_usdc REAL,
                    model_prob REAL,
                    market_price REAL,
                    edge REAL,
                    kelly_pct REAL,
                    signal_reason TEXT,
                    executed INTEGER NOT NULL DEFAULT 0,  -- 0=dry, 1=sent, 2=filled
                    dry_run INTEGER NOT NULL DEFAULT 1,
                    order_id TEXT                   -- PM order ID si ejecutado
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS bankroll_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    usdc_balance REAL NOT NULL,
                    total_exposure REAL NOT NULL DEFAULT 0.0,
                    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
                    realized_pnl_cumulative REAL NOT NULL DEFAULT 0.0,
                    n_open_positions INTEGER NOT NULL DEFAULT 0
                )
            """)
            con.commit()

    def record_order(
        self,
        side: str,
        game_date: str = None,
        game_key: str = None,
        team: str = None,
        token_id: str = None,
        price: float = None,
        shares: int = None,
        cost_usdc: float = None,
        model_prob: float = None,
        market_price: float = None,
        edge: float = None,
        kelly_pct: float = None,
        signal_reason: str = None,
        executed: int = 0,
        dry_run: bool = True,
        order_id: str = None,
    ):
        """Registra una orden (dry run o real) en la tabla orders."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT INTO orders (
                    timestamp, game_date, game_key, team, token_id,
                    side, price, shares, cost_usdc,
                    model_prob, market_price, edge, kelly_pct,
                    signal_reason, executed, dry_run, order_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, game_date, game_key, team, token_id,
                side, price, shares, cost_usdc,
                model_prob, market_price, edge, kelly_pct,
                signal_reason, executed, int(dry_run), order_id,
            ))
            con.commit()

    def open_position(
        self,
        game_date: str,
        game_key: str,
        team: str,
        token_id: str,
        entry_price: float,
        shares: int,
        model_prob: float = None,
        dry_run: bool = True,
    ):
        """Abre una nueva posicion solo si no existe ya una abierta para ese juego/equipo.

        Usa INSERT OR IGNORE para evitar sobreescribir posiciones abiertas si el predictor
        se ejecuta mas de una vez el mismo dia (entry_price y cost_basis se preservan).
        """
        cost_basis = round(shares * entry_price, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT OR IGNORE INTO positions (
                    game_date, game_key, team, token_id,
                    entry_price, shares, cost_basis, model_prob,
                    current_price, status, dry_run, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
            """, (
                game_date, game_key, team, token_id,
                entry_price, shares, cost_basis, model_prob,
                entry_price, int(dry_run), timestamp,
            ))
            con.commit()

    def close_position(
        self,
        game_date: str,
        game_key: str,
        team: str,
        exit_price: float,
        exit_reason: str,
        dry_run: bool = True,
    ):
        """Cierra una posicion abierta y calcula P&L realizado."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as con:
            row = con.execute("""
                SELECT id, entry_price, shares, cost_basis
                FROM positions
                WHERE game_date = ? AND game_key = ? AND team = ?
                  AND status = 'OPEN' AND dry_run = ?
            """, (game_date, game_key, team, int(dry_run))).fetchone()

            if not row:
                logger.warning("No open position found for %s %s %s", game_date, game_key, team)
                return

            pos_id, entry_price, shares, cost_basis = row
            realized_pnl = round(shares * (exit_price - entry_price), 2)

            con.execute("""
                UPDATE positions SET
                    exit_price = ?,
                    realized_pnl = ?,
                    exit_reason = ?,
                    status = 'CLOSED',
                    closed_at = ?
                WHERE id = ?
            """, (exit_price, realized_pnl, exit_reason, timestamp, pos_id))
            con.commit()

    def update_current_prices(self, price_map: dict[str, float]):
        """Actualiza precios actuales y P&L no realizado de posiciones abiertas.

        Args:
            price_map: {token_id: current_price}
        """
        with sqlite3.connect(self.db_path) as con:
            positions = con.execute("""
                SELECT id, token_id, entry_price, shares
                FROM positions WHERE status = 'OPEN'
            """).fetchall()

            for pos_id, token_id, entry_price, shares in positions:
                if token_id in price_map:
                    current = price_map[token_id]
                    pnl = round(shares * (current - entry_price), 2)
                    con.execute("""
                        UPDATE positions SET current_price = ?, unrealized_pnl = ?
                        WHERE id = ?
                    """, (current, pnl, pos_id))

            con.commit()

    def get_open_positions(self, dry_run: bool = True) -> list[dict]:
        """Retorna todas las posiciones abiertas."""
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("""
                SELECT * FROM positions
                WHERE status = 'OPEN' AND dry_run = ?
                ORDER BY game_date, game_key
            """, (int(dry_run),)).fetchall()
            return [dict(r) for r in rows]

    def get_total_exposure(self, dry_run: bool = True) -> float:
        """Retorna la exposicion total (suma de cost_basis de posiciones abiertas)."""
        with sqlite3.connect(self.db_path) as con:
            row = con.execute("""
                SELECT COALESCE(SUM(cost_basis), 0.0)
                FROM positions
                WHERE status = 'OPEN' AND dry_run = ?
            """, (int(dry_run),)).fetchone()
            return float(row[0])

    def get_realized_pnl(self, dry_run: bool = True) -> float:
        """Retorna el P&L realizado acumulado."""
        with sqlite3.connect(self.db_path) as con:
            row = con.execute("""
                SELECT COALESCE(SUM(realized_pnl), 0.0)
                FROM positions
                WHERE status = 'CLOSED' AND dry_run = ?
            """, (int(dry_run),)).fetchone()
            return float(row[0])

    def save_bankroll_snapshot(self, bankroll: float, dry_run: bool = True):
        """Guarda snapshot diario del bankroll."""
        today = datetime.now().strftime("%Y-%m-%d")
        positions = self.get_open_positions(dry_run)
        exposure = sum(p["cost_basis"] for p in positions)
        unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
        realized = self.get_realized_pnl(dry_run)

        with sqlite3.connect(self.db_path) as con:
            con.execute("""
                INSERT OR REPLACE INTO bankroll_history (
                    date, usdc_balance, total_exposure, unrealized_pnl,
                    realized_pnl_cumulative, n_open_positions
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (today, bankroll, exposure, unrealized, realized, len(positions)))
            con.commit()

    def print_portfolio_report(self, dry_run: bool = True):
        """Imprime reporte de portfolio actual."""
        positions = self.get_open_positions(dry_run)
        exposure = sum(p["cost_basis"] for p in positions)
        unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
        realized = self.get_realized_pnl(dry_run)

        mode = "DRY RUN" if dry_run else "LIVE"

        print(f"\n  Portfolio ({mode}): {len(positions)} open | "
              f"Exposure: ${exposure:.2f} | "
              f"Unrealized: ${unrealized:+.2f} | "
              f"Realized: ${realized:+.2f}")

        if positions:
            for p in positions:
                pnl = p.get("unrealized_pnl", 0)
                pnl_str = f"${pnl:+.2f}" if pnl else "$0.00"
                print(f"    {p['team']}: {p['shares']} shares @ ${p['entry_price']:.2f} "
                      f"-> ${p.get('current_price', p['entry_price']):.2f} "
                      f"({pnl_str})")
