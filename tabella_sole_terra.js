/* =====================================================================
   Posizione di Sole e Terra da un sito lunare, per l'app web.
   Legge tabella_sole_terra.json (generata da tabella_sole_terra.py con
   effemeridi JPL DE421) e interpola tra un'ora e l'altra.
   Stessa precisione di effemeridi.py (errore < 0.01 gradi), parallasse inclusa.

   Uso:
     const T = await caricaTabella("tabella_sole_terra.json");
     const p = soleTerra(T, -89.7748, -156.2505, 1734, new Date("2027-03-10T12:00:00Z"));
     // p.sole.el, p.sole.az, p.terra.el, p.terra.az   (gradi; az da Nord verso Est)
     // p.sole.raggio  (raggio apparente del Sole in gradi, ~0.27)
   ===================================================================== */

const R_LUNA_KM = 1737.4, R_SOLE_KM = 695700, DEG = Math.PI / 180;

async function caricaTabella(url) {
  const T = await (await fetch(url)).json();
  T.t0 = Date.parse(T.inizio);
  T.passoMs = T.passo_ore * 3600e3;
  return T;
}

// vettore (km) del corpo nel frame lunare, all'indice i della tabella
function _vett(lat, lon, dist) {
  const la = lat * DEG, lo = lon * DEG;
  return [dist * Math.cos(la) * Math.cos(lo), dist * Math.cos(la) * Math.sin(lo), dist * Math.sin(la)];
}

function _interp(T, data, corpo) {
  const f = (data.getTime() - T.t0) / T.passoMs;
  if (f < 0 || f > T.n - 1) throw new Error("Data fuori dalla tabella: " + data.toISOString());
  const i = Math.min(Math.floor(f), T.n - 2), w = f - i;
  const lat = T[corpo + "_lat"], lon = T[corpo + "_lon"];
  const d0 = corpo === "sole" ? T.sole_dist_kkm[i] * 1000 : T.terra_dist_km[i];
  const d1 = corpo === "sole" ? T.sole_dist_kkm[i + 1] * 1000 : T.terra_dist_km[i + 1];
  const a = _vett(lat[i], lon[i], d0), b = _vett(lat[i + 1], lon[i + 1], d1);
  return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
}

// elevazione e azimut di un vettore-corpo visto dal sito (lat, lon in gradi, quota in m)
function _elevAz(corpo, lat, lon, quotaM) {
  const la = lat * DEG, lo = lon * DEG;
  const su = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
  const est = [-Math.sin(lo), Math.cos(lo), 0];
  const nord = [-Math.sin(la) * Math.cos(lo), -Math.sin(la) * Math.sin(lo), Math.cos(la)];
  const r = R_LUNA_KM + (quotaM || 0) / 1000;
  const v = [corpo[0] - su[0] * r, corpo[1] - su[1] * r, corpo[2] - su[2] * r];
  const d = Math.hypot(v[0], v[1], v[2]);
  const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const el = Math.asin(dot(v, su) / d) / DEG;
  const az = (Math.atan2(dot(v, est), dot(v, nord)) / DEG + 360) % 360;
  return { el, az, dist_km: d };
}

function soleTerra(T, lat, lon, quotaM, data) {
  const s = _elevAz(_interp(T, data, "sole"), lat, lon, quotaM);
  s.raggio = Math.asin(R_SOLE_KM / s.dist_km) / DEG;
  const t = _elevAz(_interp(T, data, "terra"), lat, lon, quotaM);
  return { sole: s, terra: t };
}

// Sole/Terra visibili sopra un orizzonte di 360 valori (gradi, azimut 0..359 da Nord)
function sopraOrizzonte(corpo, orizzonte, margine = 0) {
  const n = orizzonte.length, p = corpo.az * n / 360;
  const i0 = Math.floor(p) % n, i1 = (i0 + 1) % n, w = p - Math.floor(p);
  return corpo.el - (orizzonte[i0] * (1 - w) + orizzonte[i1] * w) > margine;
}

if (typeof module !== "undefined") module.exports = { caricaTabella, soleTerra, sopraOrizzonte };
