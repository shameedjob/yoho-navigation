"""The user's home location, stored only as ciphertext.

Home is saved as {"label", "lat", "lon"} encrypted into users/{uid}.home_enc.
Coordinates are resolved once, when the user saves -- from the browser's
Geolocation API or by geocoding the typed address -- so routing later never
has to send the plaintext address to Nominatim again.

Plaintext exists only inside `load_home`, which the agent's home tool calls at
the moment it routes (agent/agent_interaction.py). Nothing returns it to the
browser: the profile page learns only whether a home is set.
"""

from __future__ import annotations

from dataclasses import dataclass

from storage import FieldCipher, UserStore

HOME_FIELD = "home"
# NYC bounding box, generous; rejects obviously wrong geolocation/geocodes.
_LAT_RANGE = (40.3, 41.2)
_LON_RANGE = (-74.5, -73.4)


@dataclass(frozen=True)
class Home:
    label: str | None
    lat: float
    lon: float


class HomeOutOfArea(ValueError):
    pass


def save_home(store: UserStore, cipher: FieldCipher, uid: str, home: Home) -> None:
    if not (_LAT_RANGE[0] <= home.lat <= _LAT_RANGE[1] and _LON_RANGE[0] <= home.lon <= _LON_RANGE[1]):
        raise HomeOutOfArea("home must be in the New York City area")
    token = cipher.encrypt_json({"label": home.label, "lat": home.lat, "lon": home.lon}, user_id=uid, field=HOME_FIELD)
    store.update_user(uid, {"home_enc": token})


def clear_home(store: UserStore, uid: str) -> None:
    store.update_user(uid, {"home_enc": None})


def has_home(user: dict | None) -> bool:
    return bool(user and user.get("home_enc"))


def load_home(store: UserStore, cipher: FieldCipher, uid: str) -> Home | None:
    user = store.get_user(uid)
    if not has_home(user):
        return None
    data = cipher.decrypt_json(user["home_enc"], user_id=uid, field=HOME_FIELD)
    return Home(label=data.get("label"), lat=float(data["lat"]), lon=float(data["lon"]))
