"""Field-level encryption for sensitive user data (home location, Google
refresh tokens, calendar event locations).

AES-256-GCM with the record it belongs to bound in as associated data
("<user_id>|<field>"), so a ciphertext copied onto another user's document, or
into another field, fails to decrypt instead of silently yielding someone
else's home address.

Keys come from YOHO_DATA_KEYS: comma-separated "<key_id>:<urlsafe-base64 32
bytes>" pairs. The first is used to encrypt; all are tried by key id on
decrypt, so a key can be rotated by prepending a new one and re-encrypting
lazily. Generate one with:

    python -m storage.crypto

Ciphertext format: "v1:<key_id>:<urlsafe-base64(nonce || ciphertext+tag)>".

The key lives in the app's environment (Secrets Manager / Secret Manager in
production), not in Firestore, so a leaked database export alone doesn't reveal
the plaintext. Upgrading to KMS envelope encryption later only changes how
the keys here are obtained.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_VERSION = "v1"
_NONCE_BYTES = 12


class DecryptionError(Exception):
    """Ciphertext is malformed, uses an unknown key, or fails authentication."""


class FieldCipher:
    def __init__(self, keys: dict[str, bytes], primary_key_id: str):
        if primary_key_id not in keys:
            raise ValueError(f"primary key id {primary_key_id!r} not in keys")
        for kid, key in keys.items():
            if ":" in kid or not kid:
                raise ValueError(f"invalid key id {kid!r}")
            if len(key) != 32:
                raise ValueError(f"key {kid!r} must be 32 bytes, got {len(key)}")
        self._keys = {kid: AESGCM(key) for kid, key in keys.items()}
        self._primary = primary_key_id

    @classmethod
    def from_env_value(cls, value: str) -> "FieldCipher":
        keys: dict[str, bytes] = {}
        primary = None
        for pair in filter(None, (p.strip() for p in value.split(","))):
            kid, _, b64 = pair.partition(":")
            keys[kid] = base64.urlsafe_b64decode(b64)
            primary = primary or kid
        if primary is None:
            raise ValueError("YOHO_DATA_KEYS is empty")
        return cls(keys, primary)

    @staticmethod
    def _aad(user_id: str, field: str) -> bytes:
        return f"{user_id}|{field}".encode()

    def encrypt(self, plaintext: str, *, user_id: str, field: str) -> str:
        nonce = secrets.token_bytes(_NONCE_BYTES)
        ct = self._keys[self._primary].encrypt(nonce, plaintext.encode(), self._aad(user_id, field))
        return f"{_VERSION}:{self._primary}:{base64.urlsafe_b64encode(nonce + ct).decode()}"

    def decrypt(self, token: str, *, user_id: str, field: str) -> str:
        try:
            version, kid, b64 = token.split(":", 2)
            blob = base64.urlsafe_b64decode(b64)
        except (ValueError, TypeError) as e:
            raise DecryptionError("malformed ciphertext") from e
        if version != _VERSION or kid not in self._keys:
            raise DecryptionError(f"unknown version or key id: {version}:{kid}")
        try:
            pt = self._keys[kid].decrypt(blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], self._aad(user_id, field))
        except Exception as e:  # cryptography raises InvalidTag
            raise DecryptionError("authentication failed") from e
        return pt.decode()

    def encrypt_json(self, value: Any, *, user_id: str, field: str) -> str:
        return self.encrypt(json.dumps(value, separators=(",", ":")), user_id=user_id, field=field)

    def decrypt_json(self, token: str, *, user_id: str, field: str) -> Any:
        return json.loads(self.decrypt(token, user_id=user_id, field=field))

    def needs_rotation(self, token: str) -> bool:
        return not token.startswith(f"{_VERSION}:{self._primary}:")


def generate_key_entry(key_id: str | None = None) -> str:
    kid = key_id or f"k{secrets.token_hex(3)}"
    return f"{kid}:{base64.urlsafe_b64encode(os.urandom(32)).decode()}"


if __name__ == "__main__":
    print(generate_key_entry())
