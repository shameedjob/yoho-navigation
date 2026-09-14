// Draws a planned route on the chat page's Leaflet map: every stop as a node,
// ride lines between them in the MTA line's color, and dashed walking lines for
// the walk from the start, transfers between rides, and the walk to the
// destination. An end with no coordinates (the user's home, which never leaves
// the server) just has no walk drawn.
//
// `route` is the /api/chat response's `route` (agent/directions.py route_legs):
//   {start, end: {kind: "place", lat, lon} | {kind: "home"} | null,
//    legs: [{type: "walk", from, to, sec}   -- from/to: a point {stop_name, lat, lon} or "start"/"destination"
//           | {type: "ride", mode, route, stops, path: [{stop_id, stop_name, lat, lon}]}]}

const YOHO_LINE_COLORS = {
  '1': '#EE352E', '2': '#EE352E', '3': '#EE352E',
  '4': '#00933C', '5': '#00933C', '6': '#00933C', '6X': '#00933C',
  '7': '#B933AD', '7X': '#B933AD',
  'A': '#0039A6', 'C': '#0039A6', 'E': '#0039A6',
  'B': '#FF6319', 'D': '#FF6319', 'F': '#FF6319', 'FX': '#FF6319', 'M': '#FF6319',
  'G': '#6CBE45',
  'J': '#996633', 'Z': '#996633',
  'L': '#A7A9AC',
  'N': '#FCCC0A', 'Q': '#FCCC0A', 'R': '#FCCC0A', 'W': '#FCCC0A',
  'S': '#808183', 'GS': '#808183', 'FS': '#808183', 'H': '#808183',
  'SI': '#0039A6', 'SIR': '#0039A6',
};
const YOHO_BUS_COLOR = '#0F6FB5';

class RouteMap {
  // map: an L.map. padding: [top, right, bottom, left] px kept clear when
  // fitting the route into view -- the chat thread covers part of the map.
  // onPlacePick(place): called when a pinned search result is clicked; return
  // false to refuse the pick (e.g. a message is still sending) and keep the pins live.
  constructor(map, { padding = [40, 40, 40, 40], onPlacePick = null } = {}) {
    this.map = map;
    this.padding = padding;
    this.onPlacePick = onPlacePick;
    if (!map.getPane('route')) {
      map.createPane('route').style.zIndex = 450; // above tiles and vector overlays, below tooltips
    }
    this.layer = L.layerGroup([], { pane: 'route' }).addTo(map);
    // Station alerts/delays sit on their own layer so they can overlay a drawn route.
    if (!map.getPane('status')) map.createPane('status').style.zIndex = 460;
    this.statusLayer = L.layerGroup([], { pane: 'status' }).addTo(map);
  }

  static lineColor(ride) {
    if (ride.mode !== 'subway') return YOHO_BUS_COLOR;
    return YOHO_LINE_COLORS[String(ride.route).toUpperCase()] || '#555';
  }

  clear() {
    this.layer.clearLayers();
    this.statusLayer.clearLayers();
  }

  // Replace whatever is drawn with `route`. Returns false (and draws nothing)
  // when there's nothing with coordinates to draw.
  show(route) {
    this.clear();
    const legs = (route && route.legs) || [];
    const rides = legs.filter(l => l.type === 'ride' && Array.isArray(l.path) && l.path.length >= 2);
    const isPoint = p => p && typeof p === 'object' && Number.isFinite(p.lat) && Number.isFinite(p.lon);
    const walks = legs.filter(l => l.type === 'walk' && isPoint(l.from) && isPoint(l.to));
    if (!rides.length && !walks.length) return false;

    const points = [];
    legs.forEach(leg => {
      if (rides.includes(leg)) this._drawRide(leg, points);
      else if (walks.includes(leg)) this._drawWalk(leg, points);
    });
    if (rides.length) this._drawEndpoints(rides[0].path[0], rides[rides.length - 1].path.at(-1));
    if (route.start && isPoint(route.start)) this._drawTripEnd(route.start, 'Start', points);
    if (route.end && isPoint(route.end)) this._drawTripEnd(route.end, 'Destination', points);
    (route.waypoints || []).filter(isPoint).forEach(w => this._drawWaypoint(w, points));

    const [top, right, bottom, left] = this.padding;
    this.map.fitBounds(L.latLngBounds(points), {
      paddingTopLeft: [left, top], paddingBottomRight: [right, bottom], maxZoom: 16,
    });
    return true;
  }

  _drawRide(ride, points) {
    const color = RouteMap.lineColor(ride);
    const latlngs = ride.path.map(s => [s.lat, s.lon]);
    points.push(...latlngs);
    // A light casing under the colored line keeps it readable on any tile.
    L.polyline(latlngs, { pane: 'route', color: '#fff', weight: 9, opacity: 0.9, lineCap: 'round', lineJoin: 'round' })
      .addTo(this.layer);
    L.polyline(latlngs, { pane: 'route', color, weight: 5, opacity: 1, lineCap: 'round', lineJoin: 'round' })
      .bindTooltip(this._rideLabel(ride), { sticky: true })
      .addTo(this.layer);
    ride.path.forEach((stop, i) => {
      const end = i === 0 || i === ride.path.length - 1;
      L.circleMarker([stop.lat, stop.lon], {
        pane: 'route', radius: end ? 6 : 3.5, color, weight: end ? 3 : 2,
        fillColor: '#fff', fillOpacity: 1,
      }).bindTooltip(stop.stop_name, { direction: 'top', offset: [0, -4] }).addTo(this.layer);
    });
  }

  // Walks are straight dashed lines: the router prices them as straight-line
  // distance, so there's no street path to draw.
  _drawWalk(leg, points) {
    const latlngs = [[leg.from.lat, leg.from.lon], [leg.to.lat, leg.to.lon]];
    points.push(...latlngs);
    const minutes = leg.sec ? ` · ${Math.max(1, Math.round(leg.sec / 60))} min` : '';
    const to = leg.to.stop_name === 'Destination' ? 'your destination' : leg.to.stop_name;
    L.polyline(latlngs, {
      pane: 'route', color: '#444', weight: 3, opacity: 0.85, dashArray: '2 7', lineCap: 'round',
    }).bindTooltip(`Walk to ${to}${minutes}`, { sticky: true }).addTo(this.layer);
  }

  // An intermediate stop of a multi-stop trip: a numbered marker.
  _drawWaypoint(stop, points) {
    points.push([stop.lat, stop.lon]);
    L.marker([stop.lat, stop.lon], {
      pane: 'route', keyboard: false, interactive: true,
      icon: L.divIcon({ className: 'yoho-waypoint', html: `<span>${stop.order}</span>`, iconSize: [24, 24], iconAnchor: [12, 12] }),
    }).bindTooltip(`Stop ${stop.order}: ${stop.name}`, { direction: 'top', offset: [0, -10] }).addTo(this.layer);
  }

  // Mark live station status from station_status: alerts ("!"), delays ("+N"
  // minutes, amber under 5, red from 5) and on-time stations (a small green dot).
  // Overlays a route drawn in the same reply; otherwise zooms to the stations.
  showStations(stations, { keepView = false } = {}) {
    this.statusLayer.clearLayers();
    const valid = (stations || []).filter(s => Number.isFinite(s.lat) && Number.isFinite(s.lon));
    if (!valid.length) return false;
    valid.forEach(s => {
      const minutes = Math.round((s.delay_sec || 0) / 60);
      const kind = s.status === 'alert' ? 'alert' : s.status === 'delayed' ? (minutes >= 5 ? 'late' : 'slow') : 'ok';
      const text = kind === 'alert' ? '!' : kind === 'ok' ? '' : `+${minutes}`;
      const size = kind === 'ok' ? 12 : 26;
      const tip = document.createElement('div');
      const title = document.createElement('strong');
      title.textContent = `${s.name} (${(s.lines || []).join(' ')})`;
      tip.appendChild(title);
      const line = t => { const d = document.createElement('div'); d.textContent = t; tip.appendChild(d); };
      if (s.status === 'ok') line('On time');
      else if (minutes > 0) line(`Next train ${minutes} min behind schedule${(s.delayed_lines || []).length ? ` (${s.delayed_lines.join(' ')})` : ''}`);
      (s.alerts || []).forEach(a => line(a));
      L.marker([s.lat, s.lon], {
        pane: 'status', keyboard: false,
        icon: L.divIcon({ className: `yoho-status yoho-status-${kind}`, html: `<span>${text}</span>`,
                          iconSize: [size, size], iconAnchor: [size / 2, size / 2] }),
      }).bindTooltip(tip, { direction: 'top', offset: [0, -size / 2] }).addTo(this.statusLayer);
    });
    if (!keepView) {
      const [top, right, bottom, left] = this.padding;
      this.map.fitBounds(L.latLngBounds(valid.map(s => [s.lat, s.lon])), {
        paddingTopLeft: [left, top], paddingBottomRight: [right, bottom], maxZoom: 15,
      });
    }
    return true;
  }

  _drawTripEnd(end, label, points) {
    points.push([end.lat, end.lon]);
    L.circleMarker([end.lat, end.lon], {
      pane: 'route', radius: 7, color: '#1b1b1b', weight: 3, fillColor: '#fff', fillOpacity: 1,
    }).bindTooltip(label, { direction: 'top', offset: [0, -6] }).addTo(this.layer);
  }

  _drawEndpoints(first, last) {
    const pin = (stop, label, fill) => L.circleMarker([stop.lat, stop.lon], {
      pane: 'route', radius: 9, color: '#fff', weight: 3, fillColor: fill, fillOpacity: 1,
    }).bindTooltip(`${label}: ${stop.stop_name}`, { direction: 'top', offset: [0, -8] }).addTo(this.layer);
    pin(first, 'Board', '#1b1b1b');
    pin(last, 'Get off', '#1b1b1b');
  }

  // Pin candidate places from a search, numbered in the order the agent lists
  // them. Replaces any drawn route. Clicking a pin picks it (onPlacePick), after
  // which this set of pins stops responding: the picked one is highlighted, the
  // rest dimmed. Returns false when there's nothing to pin.
  showPlaces(places) {
    this.clear();
    const valid = (places || []).filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon));
    if (!valid.length) return false;
    const pickable = typeof this.onPlacePick === 'function';
    const markers = [];
    const pick = (chosen) => {
      if (this.onPlacePick(chosen.place) === false) return;
      markers.forEach(m => {
        m.off('click');
        const el = m.getElement();
        if (!el) return;
        el.classList.remove('clickable');
        el.classList.add(m === chosen.marker ? 'picked' : 'dimmed');
      });
    };
    valid.forEach((place, i) => {
      const icon = L.divIcon({
        className: 'yoho-place-pin' + (pickable ? ' clickable' : ''),
        html: `<span>${i + 1}</span>`,
        iconSize: [26, 26], iconAnchor: [13, 13],
      });
      const label = document.createElement('div');
      const title = document.createElement('strong');
      title.textContent = place.name;
      label.appendChild(title);
      if (place.address) label.appendChild(document.createTextNode(` · ${place.address}`));
      if (pickable) label.appendChild(Object.assign(document.createElement('div'), { textContent: 'Click to choose' }));
      const marker = L.marker([place.lat, place.lon], { pane: 'route', icon, keyboard: false, interactive: pickable })
        .bindTooltip(label, { direction: 'top', offset: [0, -12] })
        .addTo(this.layer);
      if (pickable) marker.on('click', () => pick({ marker, place }));
      markers.push(marker);
    });
    const [top, right, bottom, left] = this.padding;
    this.map.fitBounds(L.latLngBounds(valid.map(p => [p.lat, p.lon])), {
      paddingTopLeft: [left, top], paddingBottomRight: [right, bottom], maxZoom: 15,
    });
    return true;
  }

  _rideLabel(ride) {
    const what = ride.mode === 'subway' ? `${ride.route} train` : `${ride.route} ${ride.mode}`;
    return `${what} · ${ride.stops} stop${ride.stops === 1 ? '' : 's'}`;
  }
}
