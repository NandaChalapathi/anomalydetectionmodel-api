
from fastapi.responses import JSONResponse
from fastapi import Request
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize Rate Limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

async def Rate_Limit_Handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "details": "Too many requests. Please try again later."
        }
    )