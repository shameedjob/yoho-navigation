// Shared helpers for the Yoho pages.

// fetch() against our own /api, with the header writes require (web/auth.py).
async function yohoApi(path, { method = 'GET', body } = {}) {
  const headers = { 'X-Yoho-Request': '1' };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const resp = await fetch(path, {
    method, headers, credentials: 'same-origin',
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (resp.status === 401) { window.location.href = '/login'; throw new Error('unauthenticated'); }
  let data = {};
  try { data = await resp.json(); } catch (_) { /* empty body */ }
  if (!resp.ok) { const err = new Error(data.error || resp.statusText); err.status = resp.status; err.data = data; throw err; }
  return data;
}

function yohoStoredTheme() {
  try { return localStorage.getItem('yoho-theme'); } catch (_) { return null; }
}

function yohoPrefersDark() {
  const stored = yohoStoredTheme();
  if (stored) return stored === 'dark';
  return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
}

function yohoSaveTheme(dark) {
  try { localStorage.setItem('yoho-theme', dark ? 'dark' : 'light'); } catch (_) { /* storage blocked */ }
}

// The device's location, kept in this browser only (localStorage) and sent with
// each chat message so the agent can plan trips "from here". The server uses it
// for that one request and never stores it.
const YOHO_LOCATION_KEY = 'yoho-location';
const YOHO_LOCATION_MAX_AGE_MS = 10 * 60 * 1000;  // older than this, you've probably moved

function yohoSavedLocation() {
  try {
    const loc = JSON.parse(localStorage.getItem(YOHO_LOCATION_KEY));
    if (loc && Number.isFinite(loc.lat) && Number.isFinite(loc.lon) && Date.now() - loc.ts <= YOHO_LOCATION_MAX_AGE_MS) {
      return { lat: loc.lat, lon: loc.lon };
    }
  } catch (_) { /* storage blocked or bad value */ }
  return null;
}

function yohoForgetLocation() {
  try { localStorage.removeItem(YOHO_LOCATION_KEY); } catch (_) { /* storage blocked */ }
}

// Ask for location access (the browser prompts once) and keep the saved position
// fresh while the page is open. onChange(loc | null) runs on each update; null
// means access was denied or is unavailable.
function yohoTrackLocation(onChange = () => {}) {
  if (!navigator.geolocation) { onChange(null); return; }
  navigator.geolocation.watchPosition(
    pos => {
      const loc = { lat: pos.coords.latitude, lon: pos.coords.longitude, accuracy: pos.coords.accuracy, ts: Date.now() };
      try { localStorage.setItem(YOHO_LOCATION_KEY, JSON.stringify(loc)); } catch (_) { /* storage blocked */ }
      onChange(loc);
    },
    err => {
      if (err.code === err.PERMISSION_DENIED) { yohoForgetLocation(); onChange(null); }
      // POSITION_UNAVAILABLE / TIMEOUT: keep the last saved position until it goes stale
    },
    { enableHighAccuracy: true, maximumAge: 60 * 1000, timeout: 20 * 1000 },
  );
}
