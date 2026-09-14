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
