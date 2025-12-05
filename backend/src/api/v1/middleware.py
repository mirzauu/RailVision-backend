from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class TenantContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract tenant info from token or header
        response = await call_next(request)
        return response
