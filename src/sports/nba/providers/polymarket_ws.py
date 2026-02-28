"""WebSocket real-time price feed para Polymarket CLOB.

Conecta a wss://ws-subscriptions-clob.polymarket.com para recibir
actualizaciones de precio en tiempo real. Sin auth, sin rate limits.

Uso:
    feed = PolymarketPriceFeed()
    feed.subscribe(["token_id_1", "token_id_2"])
    feed.run_in_thread()
    # ... despues de un rato:
    price = feed.get_price("token_id_1")
    feed.stop()
"""

import json
import logging
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolymarketPriceFeed:
    """WebSocket price feed para tokens de Polymarket.

    Mantiene un cache de precios en memoria actualizado en tiempo real.
    Corre en un thread separado para no bloquear el main loop.
    """

    def __init__(self, on_price_update: Callable = None):
        """
        Args:
            on_price_update: callback(token_id, price) opcional
        """
        self._prices: dict[str, float] = {}
        self._subscribed_tokens: list[str] = []
        self._ws = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._on_price_update = on_price_update
        self._lock = threading.Lock()

    def subscribe(self, token_ids: list[str]):
        """Agrega tokens a la lista de suscripcion.

        Si el WebSocket ya esta corriendo, envia el mensaje de suscripcion.
        Si no, los tokens se suscribiran cuando se llame run_in_thread().
        """
        with self._lock:
            for tid in token_ids:
                if tid and tid not in self._subscribed_tokens:
                    self._subscribed_tokens.append(tid)

        # Si ya esta conectado, suscribir inmediatamente
        if self._ws and self._running:
            self._send_subscribe(token_ids)

    def unsubscribe(self, token_ids: list[str]):
        """Remueve tokens de la suscripcion."""
        with self._lock:
            for tid in token_ids:
                if tid in self._subscribed_tokens:
                    self._subscribed_tokens.remove(tid)

    def get_price(self, token_id: str) -> float | None:
        """Retorna el ultimo precio conocido de un token."""
        with self._lock:
            return self._prices.get(token_id)

    def get_all_prices(self) -> dict[str, float]:
        """Retorna copia de todos los precios actuales."""
        with self._lock:
            return dict(self._prices)

    def run_in_thread(self):
        """Inicia el WebSocket en un thread daemon."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._thread.start()
        logger.info("Polymarket WS feed started (thread)")

    def stop(self):
        """Detiene el WebSocket y el thread."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Polymarket WS feed stopped")

    def _ws_loop(self):
        """Loop principal del WebSocket con reconnect."""
        try:
            import websocket
        except ImportError:
            logger.warning("websocket-client not installed. pip install websocket-client. WS feed disabled.")
            self._running = False
            return

        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.warning("WS connection error: %s. Reconnecting in 5s...", e)

            if self._running:
                time.sleep(5)  # Reconnect delay

    def _on_open(self, ws):
        """Callback cuando se abre la conexion."""
        logger.info("Polymarket WS connected")
        with self._lock:
            tokens = list(self._subscribed_tokens)
        if tokens:
            self._send_subscribe(tokens)

    def _send_subscribe(self, token_ids: list[str]):
        """Envia mensaje de suscripcion batch al WebSocket.

        Usa assets_ids (plural, lista) segun el formato del CLOB WS de Polymarket.
        Un unico mensaje cubre todos los tokens, evitando race conditions en conexiones
        con alta latencia donde mensajes individuales podrian perderse.
        """
        if not self._ws:
            return

        valid_ids = [tid for tid in token_ids if tid]
        if not valid_ids:
            return

        msg = json.dumps({
            "type": "market",
            "assets_ids": valid_ids,
        })
        try:
            self._ws.send(msg)
            logger.debug("Subscribed batch to %d tokens", len(valid_ids))
        except Exception as e:
            logger.warning("WS batch subscribe failed: %s", e)

    def _on_message(self, ws, message):
        """Procesa mensajes del WebSocket.

        Formato tipico del CLOB WS:
        {"event_type": "price_change", "asset_id": "...", "price": "0.65", ...}
        o
        {"event_type": "last_trade_price", "asset_id": "...", "price": "0.65"}
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        # Extraer precio segun formato del mensaje
        token_id = data.get("asset_id") or data.get("market")
        price_str = data.get("price") or data.get("last_trade_price")

        if not token_id or not price_str:
            # Puede ser un mensaje de heartbeat o tipo diferente
            return

        try:
            price = float(price_str)
        except (ValueError, TypeError):
            return

        with self._lock:
            self._prices[token_id] = price

        if self._on_price_update:
            try:
                self._on_price_update(token_id, price)
            except Exception as e:
                logger.debug("Price update callback error: %s", e)

    def _on_error(self, ws, error):
        """Callback de error."""
        logger.warning("Polymarket WS error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        """Callback de cierre."""
        logger.info("Polymarket WS closed (code=%s)", close_status_code)
