"""
Genera la tabella delle posizioni di Sole e Terra per l'app web.

Per ogni ora del periodo salva, nel sistema di riferimento lunare MOON_ME_DE421
(lo stesso di effemeridi.py e dei DEM LOLA):
    - lat/lon del punto della Luna che ha il Sole allo zenit   (punto subsolare)
    - lat/lon del punto della Luna che ha la Terra allo zenit  (punto subterrestre)
    - distanza Luna-Terra (km) e Luna-Sole (migliaia di km)

Con questi numeri il browser calcola elevazione e azimut di Sole e Terra da
qualsiasi sito (vedi tabella_sole_terra.js), con la stessa precisione di
effemeridi.py, senza Python.

Uso:  python tabella_sole_terra.py            (default: 2026-01-01 -> 2029-01-01, ogni ora)
      python tabella_sole_terra.py --inizio 2027-01-01 --anni 1 --passo-ore 1
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

from mappatura import vettori_sole_terra


def sottopunto(v):
    """Vettore (3, n) in km -> lat, lon (gradi) e distanza (km) del punto sottostante."""
    d = np.linalg.norm(v, axis=0)
    lat = np.degrees(np.arcsin(v[2] / d))
    lon = np.degrees(np.arctan2(v[1], v[0]))
    return lat, lon, d


def genera(inizio, giorni, passo_ore):
    tempi, sole, terra = vettori_sole_terra(inizio, giorni, passo_ore * 60)
    s_lat, s_lon, s_d = sottopunto(sole)
    t_lat, t_lon, t_d = sottopunto(terra)
    return {
        "formato": "tabella-sole-terra", "versione": 1,
        "descrizione": ("Punti subsolare e subterrestre sulla Luna (frame MOON_ME_DE421, "
                        "effemeridi JPL DE421 con librazione), posizioni apparenti viste dal "
                        "centro della Luna. Lon positiva a Est. Vedi tabella_sole_terra.js."),
        "inizio": inizio.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "passo_ore": passo_ore,
        "n": len(tempi),
        "sole_lat": np.round(s_lat, 4).tolist(),
        "sole_lon": np.round(s_lon, 4).tolist(),
        "sole_dist_kkm": np.round(s_d / 1000, 1).tolist(),
        "terra_lat": np.round(t_lat, 4).tolist(),
        "terra_lon": np.round(t_lon, 4).tolist(),
        "terra_dist_km": np.round(t_d, 0).astype(int).tolist(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inizio", default="2026-01-01")
    ap.add_argument("--anni", type=float, default=3)
    ap.add_argument("--passo-ore", type=float, default=1)
    ap.add_argument("--uscita", default="tabella_sole_terra.json")
    a = ap.parse_args()
    inizio = datetime.strptime(a.inizio, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    giorni = round(a.anni * 365.25)
    tab = genera(inizio, giorni, a.passo_ore)
    with open(a.uscita, "w", encoding="utf-8") as f:
        json.dump(tab, f, separators=(",", ":"))
    print(f"Salvato {a.uscita}: {tab['n']} istanti dal {tab['inizio']} ogni {a.passo_ore:g} h")
