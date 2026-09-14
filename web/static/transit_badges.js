// Turns subway and bus mentions in chat text into MTA-style badges:
//   "1 train"          -> one red badge reading "1 train"
//   "A/C/E trains"     -> blue badges "A" "C" "E", then " trains"
//   "M15-SBS bus"      -> one blue bus badge reading "M15-SBS bus"
// Needs YOHO_LINE_COLORS and YOHO_BUS_COLOR from route_map.js, loaded first.

// Longest first, so "SIR" isn't read as "S". Letters are uppercase only, so "a train" stays text.
const YOHO_TRAIN_ROUTE = '(?:SIR|SI|GS|FS|6X|7X|FX|[1-7]|[ACEBDFMGJZLNQRWSH])';
// Borough prefixes (Bx before B), express prefixes (BxM, BM, QM, SIM, X), then the number, an
// optional variant letter, and an optional Select Bus Service or limited mark.
const YOHO_BUS_ROUTE = '(?:BxM|BM|QM|SIM|X|Bx|B|M|Q|S)\\d{1,3}[A-Z]?(?:-?SBS|\\+)?';
// Separators between routes in a list: "1, 2, or 3", "A/C/E", "N & Q", "B or D".
const YOHO_ROUTE_SEP = '(?:\\s*,\\s*(?:and\\s+|or\\s+)?|\\s*/\\s*|\\s*&\\s*|\\s+(?:and|or)\\s+)';

function yohoRouteList(route) {
  return `${route}(?:${YOHO_ROUTE_SEP}${route})*`;
}

const YOHO_TRANSIT_RE = new RegExp(
  `\\b(?:(${yohoRouteList(YOHO_TRAIN_ROUTE)})(\\s+)([Tt]rains?|[Ll]ines?)` +
  `|(${yohoRouteList(YOHO_BUS_ROUTE)})(\\s+)([Bb]us(?:es)?))\\b`,
  'g'
);

// MTA prints the yellow lines' bullets with black text; everything else takes white.
const YOHO_DARK_TEXT_COLORS = new Set(['#FCCC0A']);

function yohoBadge(label, kind, route) {
  const bg = kind === 'bus'
    ? YOHO_BUS_COLOR
    : (YOHO_LINE_COLORS[route] || '#555');
  return { badge: label, kind, bg, fg: YOHO_DARK_TEXT_COLORS.has(bg) ? '#000' : '#fff' };
}

// Split `text` into [{text}] and [{badge, kind, bg, fg}] segments. Pure, so it's testable
// without a DOM.
function yohoTransitSegments(text) {
  const out = [];
  const pushText = s => {
    if (!s) return;
    const last = out[out.length - 1];
    if (last && last.text !== undefined) last.text += s; else out.push({ text: s });
  };
  let at = 0;
  YOHO_TRANSIT_RE.lastIndex = 0;
  for (let m; (m = YOHO_TRANSIT_RE.exec(text)); ) {
    const isTrain = m[1] !== undefined;
    const [list, gap, noun] = isTrain ? [m[1], m[2], m[3]] : [m[4], m[5], m[6]];
    const kind = isTrain ? 'train' : 'bus';
    const token = new RegExp(isTrain ? YOHO_TRAIN_ROUTE : YOHO_BUS_ROUTE, 'g');
    const routes = [...list.matchAll(token)];
    pushText(text.slice(at, m.index));
    if (routes.length === 1) {
      // A single route: the badge carries the noun too ("1 train").
      out.push(yohoBadge(`${list}${gap}${noun}`, kind, list));
    } else {
      // A list: badge each route, keep separators and the noun as plain text.
      let pos = 0;
      for (const r of routes) {
        pushText(list.slice(pos, r.index));
        out.push(yohoBadge(r[0], kind, r[0]));
        pos = r.index + r[0].length;
      }
      pushText(list.slice(pos) + gap + noun);
    }
    at = m.index + m[0].length;
  }
  pushText(text.slice(at));
  return out;
}

// Badge transit mentions in every text node under `root`, leaving code, links, and existing
// badges alone. Builds nodes with textContent, so it's safe on already-sanitized HTML.
function yohoDecorateTransit(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: n => (n.parentElement && n.parentElement.closest('code,pre,a,.transit-badge'))
      ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  for (let n; (n = walker.nextNode()); ) nodes.push(n);
  for (const node of nodes) {
    const segments = yohoTransitSegments(node.nodeValue);
    if (!segments.some(s => s.badge)) continue;
    const frag = document.createDocumentFragment();
    for (const s of segments) {
      if (s.text !== undefined) { frag.appendChild(document.createTextNode(s.text)); continue; }
      const span = document.createElement('span');
      span.className = `transit-badge transit-${s.kind}`;
      span.style.background = s.bg;
      span.style.color = s.fg;
      span.textContent = s.badge;
      frag.appendChild(span);
    }
    node.parentNode.replaceChild(frag, node);
  }
}
