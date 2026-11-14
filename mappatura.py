"""
Mappatura di luce, energia e comunicazione su un'AREA del polo sud lunare.

Per ogni cella di una griglia (default 500 m) dentro il DEM di un sito:
    1. calcola l'orizzonte a 360 gradi dal terreno
         - vicino (fino a 10 km): DEM del sito a 5 m/pixel
         - lontano (fino a 200 km): DEM regionale LOLA a 240 m/pixel
    2. calcola Sole e Terra visti dalla cella (effemeridi JPL DE421, come
       effemeridi.py) per tutto il periodo richiesto
    3. ricava le metriche: % luce, energia media, % comunicazione utile,
       % luce + comunicazione, buio continuo piu' lungo.

Uso (dalla cartella del repository), due modi per il terreno vicino:
    a) celle da 100 m del gruppo Terreno (site_cells.csv) - consigliato:
       python mappatura.py --celle "<...>/Sito_04/site_cells.csv" \
              --far dati_dem/LDEM_60S_240MPP_ADJ.tiff --nome Sito_04 --uscita Site04
    b) DEM a 5 m (*_surf.tif):
       python mappatura.py --dem Site01/Site01_final_adj_5mpp_surf.tif \
              --far dati_dem/LDEM_60S_240MPP_ADJ.tiff --nome Sito_01 --uscita Site01
    Su Connecting ridge i due modi differiscono in media < 1 punto percentuale
    (correlazione 0.99); il CSV e' leggermente ottimista (media su 100 m).
    Con il CSV si ottiene anche la pendenza e la classifica delle celle
    "atterrabili" (>= 50% dell'area con pendenza <= 10 gradi).

Opzioni utili: --passo 500 (m tra le celle), --inizio 2027-01-01,
--giorni 365, --passo-tempo 60 (minuti), --confronta <horizon.json del
gruppo Terreno> (verifica l'orizzonte al centro), --segna "lat,lon,nome"
(punto da evidenziare sulla mappa, ripetibile).

Convenzioni identiche a effemeridi.py: angoli in gradi, azimut da Nord in
senso orario, Luna sferica di raggio 1 737 400 m, frame MOON_ME_DE421.
Il DEM deve essere in proiezione stereografica polare SUD (prodotti PGDA/LOLA).
"""
import argparse
import base64
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.ndimage import map_coordinates

from effemeridi import (DEGRADO_POLVERE_ANNUO_PCT, MARGINE_COM_DEG,
                        PERDITA_INIZIALE_POLVERE_PCT, RAGGIO_SOLE_KM,
                        _frazione_disco, _risorse)

R_LUNA_M = 1_737_400.0
LUNA_LONLAT = f"+proj=longlat +R={R_LUNA_M} +no_defs"
CRS_POLARE = (f"+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +x_0=0 +y_0=0 "
              f"+R={R_LUNA_M} +units=m +no_defs")
PENDENZA_MAX_DEG = 10.0        # soglia di atterrabilita' (stessa del gruppo Terreno)


# --------------------------------------------------------------------------
# DEM
# --------------------------------------------------------------------------
class DEM:
    """DEM in stereografica polare sud, letto (eventualmente) solo in una finestra."""

    def __init__(self, path=None, bounds=None, z=None, transform=None, crs=None):
        if z is not None:                       # DEM gia' in memoria (es. da CSV)
            self.z, self.transform, self.crs = z, transform, crs
            self.res = abs(transform.a)
            self.bounds = rasterio.transform.array_bounds(*z.shape, transform)  # (ovest, sud, est, nord)
            self._inv = ~self.transform
            self.celle = None
            return
        self.celle = None
        with rasterio.open(path) as src:
            window = None
            if bounds is not None:
                window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
                window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            data = src.read(1, window=window, masked=True).astype("float64").filled(np.nan)
            self.z = data * src.scales[0] + src.offsets[0]
            self.transform = src.window_transform(window) if window is not None else src.transform
            self.crs = src.crs
            self.bounds = rasterio.transform.array_bounds(*self.z.shape, self.transform)  # (ovest, sud, est, nord)
            self.res = abs(self.transform.a)
        self._inv = ~self.transform

    @classmethod
    def da_celle(cls, csv_path):
        """DEM a 100 m costruito da site_cells.csv del gruppo Terreno
        (quota media di ogni cella). Conserva anche le colonne di pendenza."""
        d = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        x, y = d["x_m"].astype(float), d["y_m"].astype(float)
        passo = float(np.median(d["width_m"]))
        x0, y1 = x.min(), y.max()
        col = np.round((x - x0) / passo).astype(int)
        row = np.round((y1 - y) / passo).astype(int)
        z = np.full((row.max() + 1, col.max() + 1), np.nan)
        z[row, col] = d["elevation_mean_m"].astype(float)
        tr = rasterio.transform.from_origin(x0 - passo / 2, y1 + passo / 2, passo, passo)
        dem = cls(z=z, transform=tr, crs=rasterio.crs.CRS.from_proj4(CRS_POLARE))
        dem.celle = {"x": x, "y": y,
                     "pendenza_media": d["slope_mean_deg"].astype(float),
                     "frazione_atterrabile": d["passes_slope_filter_pct"].astype(float)}
        return dem

    def campiona(self, x, y):
        """Quota (m) nei punti x, y (m), interpolazione bilineare; NaN fuori dal DEM."""
        inv = self._inv
        col = inv.a * x + inv.b * y + inv.c - 0.5   # rasterio -> angolo pixel; scipy -> centro
        row = inv.d * x + inv.e * y + inv.f - 0.5
        h, w = self.z.shape
        dentro = (row >= 0) & (col >= 0) & (row <= h - 1) & (col <= w - 1)
        out = np.full(np.shape(x), np.nan)
        if dentro.any():
            out[dentro] = map_coordinates(self.z, [row[dentro], col[dentro]], order=1,
                                          mode="nearest", prefilter=False)
        return out


# --------------------------------------------------------------------------
# Orizzonte per ogni cella
# --------------------------------------------------------------------------
def orizzonte_celle(vicino, lontano, xs, ys, altezza_oss=2.0, n_az=360,
                    passo_vicino=10.0, dist_vicina=10_000.0,
                    passo_lontano=240.0, dist_max=200_000.0):
    """Orizzonte (gradi) a n_az azimut per ogni cella (x, y).

    Raggi tracciati come rette nel piano stereografico polare (conforme; a
    meno di 200 km dal punto l'errore di scala e' < 0.4%). Nord locale =
    direzione radiale uscente dal polo, Est = Nord ruotato di 90 gradi in
    senso orario. Curvatura lunare: termine d^2 / (2R).
    Restituisce (orizzonte[n_celle, n_az], quota_suolo[n_celle]).
    """
    az = np.radians(np.arange(n_az) * 360.0 / n_az)
    d_v = np.arange(passo_vicino, dist_vicina + 1e-6, passo_vicino)
    d_l = np.arange(dist_vicina + passo_lontano, dist_max + 1e-6, passo_lontano) if lontano else np.array([])
    dist = np.concatenate([d_v, d_l])
    caduta = dist ** 2 / (2 * R_LUNA_M)

    oriz = np.full((len(xs), n_az), np.nan)
    quota = np.full(len(xs), np.nan)
    for k, (x0, y0) in enumerate(zip(xs, ys)):
        rho = np.hypot(x0, y0)
        nx, ny = x0 / rho, y0 / rho          # Nord locale (lontano dal polo)
        ex, ey = ny, -nx                     # Est locale
        z0 = vicino.campiona(np.array([x0]), np.array([y0]))[0]
        if not np.isfinite(z0) and lontano:
            z0 = lontano.campiona(np.array([x0]), np.array([y0]))[0]
        if not np.isfinite(z0):
            continue
        quota[k] = z0
        dx = np.cos(az)[:, None] * nx + np.sin(az)[:, None] * ex
        dy = np.cos(az)[:, None] * ny + np.sin(az)[:, None] * ey
        px = x0 + dx * dist[None, :]
        py = y0 + dy * dist[None, :]
        h = np.full(px.shape, np.nan)
        nv = len(d_v)
        h[:, :nv] = vicino.campiona(px[:, :nv], py[:, :nv])
        if lontano:
            buchi = ~np.isfinite(h)
            h[buchi] = lontano.campiona(px[buchi], py[buchi])
        ang = np.degrees(np.arctan2(h - z0 - altezza_oss - caduta[None, :], dist[None, :]))
        oriz[k] = np.nanmax(np.where(np.isfinite(ang), ang, -90.0), axis=1)
    return oriz, quota


# --------------------------------------------------------------------------
# Sole e Terra nel frame lunare, una volta per tutte le celle
# --------------------------------------------------------------------------
def vettori_sole_terra(inizio, giorni, passo_min):
    """Posizioni apparenti (km) di Sole e Terra viste dal centro della Luna,
    nel frame MOON_ME_DE421: array (3, n_tempi). Restituisce anche i tempi."""
    r = _risorse()
    ts, eph, frame = r['ts'], r['eph'], r['frame']
    n = int(giorni * 24 * 60 / passo_min) + 1
    tempi = [inizio + timedelta(minutes=passo_min * k) for k in range(n)]
    t = ts.from_datetimes(tempi)
    luna = eph['moon'].at(t)
    sole = luna.observe(eph['sun']).apparent().frame_xyz(frame).km
    terra = luna.observe(eph['earth']).apparent().frame_xyz(frame).km
    return tempi, np.asarray(sole), np.asarray(terra)


def _assi_locali(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    su = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    est = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)])
    nord = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
    return su, est, nord


def elev_az(corpo_km, lat, lon, quota_m):
    """Elevazione e azimut (gradi) di un corpo per ogni cella e ogni istante:
    array (n_celle, n_tempi). Include la parallasse (posizione della cella)."""
    su, est, nord = _assi_locali(lat, lon)
    p = su * ((R_LUNA_M + np.nan_to_num(quota_m)) / 1000.0)        # (3, nc) km
    v = corpo_km[:, None, :] - p[:, :, None]                         # (3, nc, nt)
    dist = np.linalg.norm(v, axis=0)
    el = np.degrees(np.arcsin(np.einsum('ict,ic->ct', v, su) / dist))
    az = np.degrees(np.arctan2(np.einsum('ict,ic->ct', v, est),
                               np.einsum('ict,ic->ct', v, nord))) % 360.0
    return el, az, dist


def _interp_orizzonte(oriz, az):
    """Orizzonte di ogni cella interpolato agli azimut az (n_celle, n_tempi)."""
    n_az = oriz.shape[1]
    pos = az * n_az / 360.0
    i0 = np.floor(pos).astype(int) % n_az
    i1 = (i0 + 1) % n_az
    f = pos - np.floor(pos)
    righe = np.arange(oriz.shape[0])[:, None]
    return oriz[righe, i0] * (1 - f) + oriz[righe, i1] * f


def _sequenza_max(b, passo_min):
    """Ore della sequenza di True consecutivi piu' lunga, per ogni riga."""
    out = np.zeros(b.shape[0])
    for i, riga in enumerate(b):
        if not riga.any():
            continue
        d = np.diff(np.concatenate([[0], riga.view(np.int8), [0]]))
        inizi, fini = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        out[i] = (fini - inizi).max() * passo_min / 60.0
    return out


# --------------------------------------------------------------------------
# Metriche per cella
# --------------------------------------------------------------------------
def metriche_celle(oriz, lat, lon, quota, sole, terra, passo_min, blocco=48,
                   margine_com_deg=MARGINE_COM_DEG,
                   perdita_iniziale_polvere_pct=PERDITA_INIZIALE_POLVERE_PCT,
                   degrado_polvere_annuo_pct=DEGRADO_POLVERE_ANNUO_PCT):
    nc, nt = len(lat), sole.shape[1]
    giorni = np.arange(nt) * passo_min / 1440.0
    polvere = ((1 - perdita_iniziale_polvere_pct / 100)
               * np.clip(1 - degrado_polvere_annuo_pct / 100 * giorni / 365.25, 0, 1))
    ris = {k: np.full(nc, np.nan) for k in
           ("luce_pct", "energia_pct", "comunicazione_pct", "luce_e_com_pct",
            "buio_max_h", "senza_com_max_h")}
    for a in range(0, nc, blocco):
        s = slice(a, min(a + blocco, nc))
        ok = np.isfinite(quota[s])
        if not ok.any():
            continue
        idx = np.arange(a, min(a + blocco, nc))[ok]
        el_s, az_s, d_s = elev_az(sole, lat[idx], lon[idx], quota[idx])
        el_t, az_t, _ = elev_az(terra, lat[idx], lon[idx], quota[idx])
        marg_s = el_s - _interp_orizzonte(oriz[idx], az_s)
        marg_t = el_t - _interp_orizzonte(oriz[idx], az_t)
        r_sole = np.degrees(np.arcsin(RAGGIO_SOLE_KM / d_s))
        frazione = _frazione_disco(marg_s / r_sole)
        luce = marg_s > 0
        com = marg_t > margine_com_deg
        ris["luce_pct"][idx] = 100 * luce.mean(axis=1)
        ris["energia_pct"][idx] = 100 * (frazione * polvere[None, :]).mean(axis=1)
        ris["comunicazione_pct"][idx] = 100 * com.mean(axis=1)
        ris["luce_e_com_pct"][idx] = 100 * (luce & com).mean(axis=1)
        ris["buio_max_h"][idx] = _sequenza_max(~luce, passo_min)
        ris["senza_com_max_h"][idx] = _sequenza_max(~com, passo_min)
    return ris


def evoluzione_celle(oriz, lat, lon, quota, sole, terra, blocco=48,
                     margine_com_deg=MARGINE_COM_DEG):
    """Per ogni istante e ogni cella: Sole visibile? Terra utile? (array bool
    [n_tempi, n_celle]). Serve per lo slider temporale dell'app."""
    nc, nt = len(lat), sole.shape[1]
    luce = np.zeros((nt, nc), bool)
    com = np.zeros((nt, nc), bool)
    for a in range(0, nc, blocco):
        idx = np.arange(a, min(a + blocco, nc))
        idx = idx[np.isfinite(quota[idx])]
        if not len(idx):
            continue
        el_s, az_s, _ = elev_az(sole, lat[idx], lon[idx], quota[idx])
        el_t, az_t, _ = elev_az(terra, lat[idx], lon[idx], quota[idx])
        luce[:, idx] = (el_s - _interp_orizzonte(oriz[idx], az_s) > 0).T
        com[:, idx] = (el_t - _interp_orizzonte(oriz[idx], az_t) > margine_com_deg).T
    return luce, com


def _bit_b64(b):
    return base64.b64encode(np.packbits(b.ravel()).tobytes()).decode("ascii")


# --------------------------------------------------------------------------
# Programma principale
# --------------------------------------------------------------------------
def mappa_sito(dem_path=None, far_path=None, nome=None, passo=500.0, inizio=None,
               giorni=365, passo_tempo=60, n_az=360, margine_bordo=0.0,
               celle_csv=None, punti=None, passo_evoluzione=360):
    t0 = time.time()
    vicino = DEM.da_celle(celle_csv) if celle_csv else DEM(dem_path)
    sx0, sy0, sx1, sy1 = vicino.bounds
    lontano = None
    if far_path:
        m = 205_000.0
        lontano = DEM(far_path, bounds=(sx0 - m, sy0 - m, sx1 + m, sy1 + m))

    # griglia di celle (centri) dentro il DEM del sito
    gx = np.arange(sx0 + passo / 2 + margine_bordo, sx1 - margine_bordo, passo)
    gy = np.arange(sy1 - passo / 2 - margine_bordo, sy0 + margine_bordo, -passo)
    X, Y = np.meshgrid(gx, gy)
    xs, ys = X.ravel(), Y.ravel()
    a_lonlat = Transformer.from_crs(vicino.crs, LUNA_LONLAT, always_xy=True)
    lon, lat = a_lonlat.transform(xs, ys)
    lon, lat = np.asarray(lon), np.asarray(lat)
    print(f"[{nome}] {len(xs)} celle ({len(gx)} x {len(gy)}), passo {passo:.0f} m")

    oriz, quota = orizzonte_celle(vicino, lontano, xs, ys, n_az=n_az)
    print(f"[{nome}] orizzonti calcolati in {time.time() - t0:.0f} s")

    inizio = inizio or datetime(2027, 1, 1, tzinfo=timezone.utc)
    tempi, sole, terra = vettori_sole_terra(inizio, giorni, passo_tempo)
    ris = metriche_celle(oriz, lat, lon, quota, sole, terra, passo_tempo)
    print(f"[{nome}] metriche calcolate, totale {time.time() - t0:.0f} s")

    evoluzione = None
    if passo_evoluzione:
        tempi_e, sole_e, terra_e = vettori_sole_terra(inizio, giorni, passo_evoluzione)
        luce_e, com_e = evoluzione_celle(oriz, lat, lon, quota, sole_e, terra_e)
        evoluzione = {
            "formato": "evoluzione-sito-lunare", "versione": 1, "nome": nome,
            "inizio": inizio.strftime('%Y-%m-%dT%H:%M:%SZ'), "passo_min": passo_evoluzione,
            "n_istanti": len(tempi_e), "forma_griglia": [len(gy), len(gx)],
            "codifica": ("base64 di bit impacchettati (numpy.packbits, primo bit = bit piu' "
                         "significativo del byte); 1 = vero. Ordine: istante, riga, colonna. "
                         "Bit k -> istante k // (righe*colonne), cella k % (righe*colonne)."),
            "luce": _bit_b64(luce_e), "comunicazione": _bit_b64(com_e),
        }
        print(f"[{nome}] evoluzione temporale: {len(tempi_e)} istanti ogni {passo_evoluzione:g} min")

    forma = (len(gy), len(gx))
    extra = {}
    if vicino.celle is not None:                 # pendenza dal CSV, aggregata sulle celle
        c = vicino.celle
        ci = np.floor((c["x"] - (gx[0] - passo / 2)) / passo).astype(int)
        ri = np.floor(((gy[0] + passo / 2) - c["y"]) / passo).astype(int)
        ok = (ci >= 0) & (ci < len(gx)) & (ri >= 0) & (ri < len(gy))
        for chiave, col in (("pendenza_media_deg", "pendenza_media"),
                            ("area_pendenza_le10_pct", "frazione_atterrabile")):
            somma, num = np.zeros(forma), np.zeros(forma)
            np.add.at(somma, (ri[ok], ci[ok]), c[col][ok])
            np.add.at(num, (ri[ok], ci[ok]), 1)
            extra[chiave] = np.where(num > 0, somma / np.maximum(num, 1), np.nan).ravel()
    atterrabile = (extra["area_pendenza_le10_pct"] >= 50) if extra else np.ones(len(xs), bool)
    migliori = _migliori_celle(ris, atterrabile, xs, ys, lat, lon, quota, extra)

    return {
        "formato": "mappa-sito-lunare", "versione": 1,
        "nome": nome or Path(dem_path).stem,
        "griglia": {"crs": "stereografica polare sud, R=1737400 m, lon_0=0",
                    "x_m": gx.tolist(), "y_m": gy.tolist(), "passo_m": passo,
                    "forma": list(forma), "ordine": "righe da nord-griglia (y max) a y min"},
        "periodo": {"inizio": inizio.strftime('%Y-%m-%dT%H:%M:%SZ'), "giorni": giorni,
                    "passo_min": passo_tempo},
        "ipotesi": {"altezza_osservatore_m": 2.0, "n_azimut": n_az,
                    "dem_vicino": Path(celle_csv or dem_path).name,
                    "atterrabile": (f">= 50% dell'area della cella con pendenza <= {PENDENZA_MAX_DEG:.0f} gradi"
                                    if extra else "non valutato (serve site_cells.csv)"),
                    "dem_lontano": Path(far_path).name if far_path else None,
                    "distanza_vicina_m": 10_000, "distanza_massima_m": 200_000 if far_path else 10_000,
                    "margine_com_deg": MARGINE_COM_DEG,
                    "degrado_polvere_annuo_pct": DEGRADO_POLVERE_ANNUO_PCT,
                    "fonti": "FONTI.txt"},
        "layers": {
            "lat": np.round(lat.reshape(forma), 6).tolist(),
            "lon": np.round(lon.reshape(forma), 6).tolist(),
            "quota_m": np.round(quota.reshape(forma), 1).tolist(),
            "orizzonte_max_deg": np.round(np.nanmax(oriz, axis=1).reshape(forma), 2).tolist(),
            **{k: np.round(v.reshape(forma), 2).tolist() for k, v in ris.items()},
            **{k: np.round(v.reshape(forma), 2).tolist() for k, v in extra.items()},
        },
        "migliori_celle": migliori,
        "punti_segnati": punti or [],
        "_evoluzione": evoluzione,
        "_orizzonti": oriz,     # non salvato nel JSON (troppo grande), utile per i controlli
    }


def _migliori_celle(ris, atterrabile, xs, ys, lat, lon, quota, extra, n=5, distanza_min=1000.0):
    """Le n celle atterrabili con piu' ore di luce + comunicazione (poi energia),
    distanti tra loro almeno distanza_min metri."""
    chiave = np.where(atterrabile & np.isfinite(ris["luce_e_com_pct"]),
                      ris["luce_e_com_pct"] + 1e-3 * np.nan_to_num(ris["energia_pct"]), -np.inf)
    scelte = []
    for i in np.argsort(-chiave):
        if not np.isfinite(chiave[i]) or len(scelte) == n:
            break
        if all(np.hypot(xs[i] - xs[j], ys[i] - ys[j]) >= distanza_min for j in scelte):
            scelte.append(i)
    out = []
    for i in scelte:
        r = {"lat": round(float(lat[i]), 5), "lon": round(float(lon[i]), 5),
             "x_m": float(xs[i]), "y_m": float(ys[i]), "quota_m": round(float(quota[i]), 1)}
        for k in ("luce_pct", "comunicazione_pct", "luce_e_com_pct", "energia_pct", "buio_max_h", "senza_com_max_h"):
            r[k] = round(float(ris[k][i]), 1)
        for k in extra:
            r[k] = round(float(extra[k][i]), 1)
        out.append(r)
    return out


def _json_pulito(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, list):
        return [_json_pulito(v) for v in o]
    if isinstance(o, dict):
        return {k: _json_pulito(v) for k, v in o.items() if not k.startswith('_')}
    return o


def salva(mappa, cartella):
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    nome = mappa["nome"]
    p_json = cartella / f"mappa_{nome}.json"
    p_json.write_text(json.dumps(_json_pulito(mappa), separators=(",", ":")), encoding="utf-8")
    disegna(mappa, cartella / f"mappa_{nome}.png")
    if mappa.get("_evoluzione"):
        (cartella / f"evoluzione_{nome}.json").write_text(
            json.dumps(mappa["_evoluzione"], separators=(",", ":")), encoding="utf-8")
    return p_json


def disegna(mappa, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g, L = mappa["griglia"], mappa["layers"]
    p = g["passo_m"] / 2
    ext = [(g["x_m"][0] - p) / 1000, (g["x_m"][-1] + p) / 1000,
           (g["y_m"][-1] - p) / 1000, (g["y_m"][0] + p) / 1000]
    lc = np.array(L["luce_e_com_pct"], dtype=float)
    if "area_pendenza_le10_pct" in L:
        lc_att = np.where(np.array(L["area_pendenza_le10_pct"], dtype=float) >= 50, lc, np.nan)
    else:
        lc_att = lc
    pannelli = [("quota_m", None, "Quota [m]", "gist_earth", None),
                ("pendenza_media_deg", None, "Pendenza media [gradi]", "RdYlGn_r", (0, 25)),
                ("luce_pct", None, "Luce [% del tempo]", "inferno", (0, 100)),
                ("comunicazione_pct", None, "Comunicazione utile [%]", "viridis", (0, 100)),
                ("luce_e_com_pct", None, "Luce + comunicazione [%]", "magma", (0, 100)),
                ("energia_pct", None, "Energia media [% del massimo]", "YlOrBr_r", (0, 100)),
                ("buio_max_h", None, "Buio continuo piu' lungo [ore]", "Blues", None),
                (None, lc_att, "Luce + com. solo dove atterrabile [%]", "magma", (0, 100))]
    fig, axs = plt.subplots(2, 4, figsize=(20, 9.5))
    for ax, (k, arr, titolo, cmap, lim) in zip(axs.ravel(), pannelli):
        if arr is None:
            if k not in L:
                ax.axis("off")
                ax.set_title(titolo + "\n(non disponibile)")
                continue
            arr = np.array(L[k], dtype=float)
        if k is None:
            ax.set_facecolor("#cfcfcf")
        im = ax.imshow(arr, extent=ext, cmap=cmap, vmin=lim[0] if lim else None,
                       vmax=lim[1] if lim else None, origin="upper")
        ax.set_title(titolo)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        fig.colorbar(im, ax=ax, shrink=0.8)
        if k in ("luce_e_com_pct", None):
            for n, c in enumerate(mappa.get("migliori_celle", []), 1):
                ax.plot(c["x_m"] / 1000, c["y_m"] / 1000, "o", ms=11, mfc="none", mec="cyan", mew=2)
                ax.text(c["x_m"] / 1000, c["y_m"] / 1000, f" {n}", color="cyan", fontsize=10, weight="bold")
        for pt in mappa.get("punti_segnati", []):
            ax.plot(pt["x_m"] / 1000, pt["y_m"] / 1000, "*", ms=14, mfc="white", mec="black")
            ax.text(pt["x_m"] / 1000, pt["y_m"] / 1000, " " + pt["nome"], color="white", fontsize=8,
                    weight="bold", va="bottom")
    per = mappa["periodo"]
    fig.suptitle(f"{mappa['nome']} - {per['inizio'][:10]} + {per['giorni']:g} giorni "
                 f"(celle {g['passo_m']:.0f} m, terreno fino a {mappa['ipotesi']['distanza_massima_m'] / 1000:.0f} km)"
                 f" - cerchi azzurri: 5 migliori celle atterrabili, stelle: punti di riferimento")
    fig.tight_layout()
    fig.savefig(file, dpi=100)
    plt.close(fig)


def confronta_con_terreno(mappa, horizon_json, dem_path, far_path, celle_csv=None):
    """Calcola l'orizzonte nel punto del file del gruppo Terreno e lo confronta."""
    h = json.loads(Path(horizon_json).read_text(encoding="utf-8"))
    vicino = DEM.da_celle(celle_csv) if celle_csv else DEM(dem_path)
    t = Transformer.from_crs(LUNA_LONLAT, vicino.crs, always_xy=True)
    x, y = t.transform(h["lon"], h["lat"])
    lontano = None
    if far_path:
        m = 205_000.0
        lontano = DEM(far_path, bounds=(x - m, y - m, x + m, y + m))
    nostro, _ = orizzonte_celle(vicino, lontano, np.array([x]), np.array([y]))
    diff = nostro[0] - np.array(h["orizzonte"])
    print(f"Confronto orizzonte con {Path(horizon_json).name}: differenza media {np.mean(diff):+.3f} gradi, "
          f"media assoluta {np.mean(np.abs(diff)):.3f}, massima {np.max(np.abs(diff)):.3f}")
    return diff


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mappa luce/comunicazione su un'area lunare")
    ap.add_argument("--dem", help="DEM del sito (es. *_surf.tif, 5 m/pixel)")
    ap.add_argument("--celle", help="site_cells.csv del gruppo Terreno (alternativa a --dem)")
    ap.add_argument("--far", help="DEM regionale per il terreno lontano (es. LDEM_60S_240MPP_ADJ.TIF)")
    ap.add_argument("--nome", help="nome del sito")
    ap.add_argument("--passo", type=float, default=500.0, help="distanza tra le celle in metri")
    ap.add_argument("--inizio", default="2027-01-01", help="data di inizio (AAAA-MM-GG, UTC)")
    ap.add_argument("--giorni", type=float, default=365)
    ap.add_argument("--passo-tempo", type=float, default=60, help="minuti tra due istanti")
    ap.add_argument("--uscita", default="output/mappe")
    ap.add_argument("--passo-evoluzione", type=float, default=360,
                    help="minuti tra i fotogrammi per lo slider temporale (0 = non salvare)")
    ap.add_argument("--confronta", help="horizon.json del gruppo Terreno da confrontare")
    ap.add_argument("--segna", action="append", default=[],
                    help='punto da evidenziare: "lat,lon,nome" (ripetibile)')
    a = ap.parse_args()
    if not (a.dem or a.celle):
        ap.error("serve --dem oppure --celle")

    a_xy = Transformer.from_crs(LUNA_LONLAT, CRS_POLARE, always_xy=True)
    punti = []
    for s_ in a.segna:
        la, lo, *nome = s_.split(",")
        x, y = a_xy.transform(float(lo), float(la))
        punti.append({"lat": float(la), "lon": float(lo), "nome": ",".join(nome).strip() or "punto",
                      "x_m": float(x), "y_m": float(y)})

    inizio = datetime.strptime(a.inizio, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    m = mappa_sito(a.dem, a.far, a.nome, a.passo, inizio, a.giorni, a.passo_tempo,
                   celle_csv=a.celle, punti=punti, passo_evoluzione=a.passo_evoluzione)
    print("Salvato:", salva(m, a.uscita))
    print("Migliori celle atterrabili (luce + comunicazione):")
    for n, c in enumerate(m["migliori_celle"], 1):
        print(f"  {n}. lat {c['lat']:.5f} lon {c['lon']:.5f} quota {c['quota_m']:.0f} m | "
              f"luce {c['luce_pct']}% com {c['comunicazione_pct']}% L+C {c['luce_e_com_pct']}% "
              f"energia {c['energia_pct']}% buio max {c['buio_max_h']:.0f} h"
              + (f" | pendenza {c['pendenza_media_deg']} gradi" if "pendenza_media_deg" in c else ""))
    if a.confronta:
        confronta_con_terreno(m, a.confronta, a.dem, a.far, a.celle)
