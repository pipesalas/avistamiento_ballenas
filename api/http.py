"""HTTP caching for the public dataset, and compression for API responses."""

from starlette.datastructures import MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PublicDataMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.compressed = GZipMiddleware(app, minimum_size=1000, compresslevel=5)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in ("GET", "HEAD"):
            await self.app(scope, receive, send)
            return

        is_api = scope["path"].startswith("/api/")
        is_media = scope["path"].startswith("/media/")

        async def send_cached(message: Message) -> None:
            if message["type"] == "http.response.start" and message["status"] in (200, 304):
                headers = MutableHeaders(scope=message)
                if is_api:
                    headers.setdefault("Cache-Control", "public, max-age=300, s-maxage=3600")
                elif is_media:
                    headers.setdefault("Cache-Control", "public, max-age=86400")
            await send(message)

        # Photos are already compressed; recompressing them wastes CPU.
        target = self.compressed if is_api else self.app
        await target(scope, receive, send_cached)
