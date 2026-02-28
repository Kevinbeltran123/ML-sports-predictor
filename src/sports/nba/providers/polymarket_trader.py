"""Polymarket order execution via py-clob-client SDK.

Dry-run por defecto. Solo ejecuta ordenes reales con --execute flag.
Requiere POLYMARKET_PRIVATE_KEY en .env (Polygon wallet private key).

Instalacion:
  pip install py-clob-client

Uso:
  trader = PolymarketTrader()           # dry_run=True default
  trader.buy_shares(token_id, 0.60, 50) # log only
  trader = PolymarketTrader(dry_run=False)  # real trading
  trader.buy_shares(token_id, 0.60, 50)     # sends to CLOB
"""

import logging
import os

logger = logging.getLogger(__name__)

# CLOB API endpoints
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


class PolymarketTrader:
    """Ejecuta ordenes en Polymarket CLOB.

    Attributes:
        dry_run: si True, solo loguea ordenes sin ejecutar
        client: py-clob-client ClobClient (None si dry_run o no disponible)
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.client = None

        if not dry_run:
            self._init_client()

    def _init_client(self):
        """Inicializa el py-clob-client con la private key."""
        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not set. Falling back to dry-run.")
            self.dry_run = True
            return

        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            self.client = ClobClient(
                host=CLOB_HOST,
                key=private_key,
                chain_id=CHAIN_ID,
            )

            # Derive API credentials (creates/gets API key from CLOB)
            try:
                creds = self.client.derive_api_key()
                self.client.set_api_creds(creds)
                logger.info("Polymarket CLOB client initialized (Polygon)")
            except Exception as e:
                logger.warning("API key derivation failed: %s. Using basic auth.", e)

        except ImportError:
            logger.warning("py-clob-client not installed. pip install py-clob-client. Falling back to dry-run.")
            self.dry_run = True
        except Exception as e:
            logger.error("Failed to init Polymarket client: %s. Falling back to dry-run.", e)
            self.dry_run = True

    def buy_shares(
        self,
        token_id: str,
        price: float,
        size: int,
        order_type: str = "GTC",
    ) -> dict:
        """Compra shares de un token.

        Args:
            token_id: CLOB token ID del outcome
            price: precio limite (0-1)
            size: numero de shares
            order_type: "GTC" (Good Till Cancel) o "FOK" (Fill or Kill)

        Returns:
            dict con order_id si ejecutado, o info de dry-run
        """
        cost = round(size * price, 2)

        if self.dry_run:
            logger.info("[DRY RUN] BUY %d shares @ $%.2f = $%.2f (token: %s)",
                        size, price, cost, token_id[:16] if token_id else "N/A")
            return {"dry_run": True, "side": "BUY", "price": price, "size": size, "cost": cost}

        try:
            from py_clob_client.order_builder.constants import BUY

            order = self.client.create_order(
                order_args={
                    "token_id": token_id,
                    "price": price,
                    "size": size,
                    "side": BUY,
                },
            )
            result = self.client.post_order(order, order_type=order_type)
            logger.info("BUY order placed: %d shares @ $%.2f = $%.2f | order_id=%s",
                        size, price, cost, result.get("orderID", "?"))
            return {"dry_run": False, "order_id": result.get("orderID"), **result}

        except Exception as e:
            logger.error("BUY order failed: %s", e)
            return {"error": str(e), "dry_run": False}

    def sell_shares(
        self,
        token_id: str,
        price: float,
        size: int,
        order_type: str = "GTC",
    ) -> dict:
        """Vende shares de un token.

        Args:
            token_id: CLOB token ID
            price: precio limite de venta (0-1)
            size: numero de shares a vender
            order_type: "GTC" o "FOK"

        Returns:
            dict con order_id si ejecutado
        """
        proceeds = round(size * price, 2)

        if self.dry_run:
            logger.info("[DRY RUN] SELL %d shares @ $%.2f = $%.2f (token: %s)",
                        size, price, proceeds, token_id[:16] if token_id else "N/A")
            return {"dry_run": True, "side": "SELL", "price": price, "size": size, "proceeds": proceeds}

        try:
            from py_clob_client.order_builder.constants import SELL

            order = self.client.create_order(
                order_args={
                    "token_id": token_id,
                    "price": price,
                    "size": size,
                    "side": SELL,
                },
            )
            result = self.client.post_order(order, order_type=order_type)
            logger.info("SELL order placed: %d shares @ $%.2f = $%.2f | order_id=%s",
                        size, price, proceeds, result.get("orderID", "?"))
            return {"dry_run": False, "order_id": result.get("orderID"), **result}

        except Exception as e:
            logger.error("SELL order failed: %s", e)
            return {"error": str(e), "dry_run": False}

    def cancel_order(self, order_id: str) -> bool:
        """Cancela una orden abierta."""
        if self.dry_run:
            logger.info("[DRY RUN] CANCEL order %s", order_id)
            return True

        try:
            self.client.cancel(order_id)
            logger.info("Order cancelled: %s", order_id)
            return True
        except Exception as e:
            logger.error("Cancel failed for %s: %s", order_id, e)
            return False

    def cancel_all(self) -> bool:
        """Cancela todas las ordenes abiertas."""
        if self.dry_run:
            logger.info("[DRY RUN] CANCEL ALL orders")
            return True

        try:
            self.client.cancel_all()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error("Cancel all failed: %s", e)
            return False

    def get_open_orders(self) -> list:
        """Retorna ordenes abiertas del usuario."""
        if self.dry_run:
            return []

        try:
            return self.client.get_orders() or []
        except Exception as e:
            logger.error("Failed to get open orders: %s", e)
            return []
