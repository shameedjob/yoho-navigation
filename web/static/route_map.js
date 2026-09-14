// Draws a planned route on the chat page's Leaflet map: every stop as a node,
// ride lines between them in the MTA line's color, and dashed walking lines
// for transfers between rides. The walks from the start and to the destination
// aren't drawn yet.
//
// `route` is the /api/chat response's `route` (agent/directions.py route_legs):
//   {legs: [{type: "walk", from?, to, sec} |
//           {type: "ride", mode, route, stops, path: [{stop_id, stop_name, lat, lon}]}]}

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
  constructor(map, { padding = [40, 40, 40, 40] } = {}) {
    this.map = map;
    this.padding = padding;
    if (!map.getPane('route')) {
      map.createPane('route').style.zIndex = 450; // above the tint pane, below markers' popups
    }
    this.layer = L.layerGroup([], { pane: 'route' }).addTo(map);
  }

  static lineColor(ride) {
    if (ride.mode !== 'subway') return YOHO_BUS_COLOR;
    return YOHO_LINE_COLORS[String(ride.route).toUpperCase()] || '#555';
  }

  clear() {
    this.layer.clearLayers();
  }

  // Replace whatever is drawn with `route`. Returns false (and draws nothing)
  // when the route has no ride with coordinates.
  show(route) {
    this.clear();
    const legs = (route && route.legs) || [];
    const rides = legs.filter(l => l.type === 'ride' && Array.isArray(l.path) && l.path.length >= 2);
    if (!rides.length) return false;

    const points = [];
    legs.forEach(leg => {
      if (leg.type === 'ride' && rides.includes(leg)) {
        this._drawRide(leg, points);
      } else if (leg.type === 'walk' && typeof leg.from === 'object' && typeof leg.to === 'object') {
        this._drawTransferWalk(leg);  // between two rides; the first and last walks have no from/to stop
      }
    });
    this._drawEndpoints(rides[0].path[0], rides[rides.length - 1].path.at(-1));

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

  _drawTransferWalk(leg) {
    L.polyline([[leg.from.lat, leg.from.lon], [leg.to.lat, leg.to.lon]], {
      pane: 'route', color: '#444', weight: 3, opacity: 0.85, dashArray: '2 7', lineCap: 'round',
    }).bindTooltip(`Walk to ${leg.to.stop_name}`, { sticky: true }).addTo(this.layer);
  }

  _drawEndpoints(first, last) {
    const pin = (stop, label, fill) => L.circleMarker([stop.lat, stop.lon], {
      pane: 'route', radius: 9, color: '#fff', weight: 3, fillColor: fill, fillOpacity: 1,
    }).bindTooltip(`${label}: ${stop.stop_name}`, { direction: 'top', offset: [0, -8] }).addTo(this.layer);
    pin(first, 'Board', '#1b1b1b');
    pin(last, 'Get off', '#1b1b1b');
  }

  _rideLabel(ride) {
    const what = ride.mode === 'subway' ? `${ride.route} train` : `${ride.route} ${ride.mode}`;
    return `${what} · ${ride.stops} stop${ride.stops === 1 ? '' : 's'}`;
  }
}
