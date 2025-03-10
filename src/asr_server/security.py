from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from asr_server.dependencies import get_config

auth_scheme = HTTPBearer(scheme_name="API key")
api_key = get_config().api_key  # HACK

if not api_key:  # HACK

    def check_api_key() -> None:
        # if No API key is set, so we don't check for it.
        pass
else:

    def check_api_key(api_key: Annotated[HTTPAuthorizationCredentials, Depends(auth_scheme)]) -> str:
        if api_key.scheme != "Bearer":
            raise HTTPException(status_code=403, detail="Invalid authentication scheme")
        if api_key.credentials != api_key:
            raise HTTPException(status_code=403, detail="Unauthorized")

        return api_key.credentials
