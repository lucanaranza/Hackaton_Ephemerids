"""
Motore delle effemeridi per la valutazione dei siti di atterraggio lunari.

Funzione principale:
    valuta_sito(lat, lon, inizio, giorni=30, passo_min=10, orizzonte=None)

Restituisce un dizionario (convertibile in JSON) con:
    - "serie":    istanti con elevazione/azimut di Sole e Terra
    - "eventi":   alba/tramonto del Sole, sorgere/tramonto della Terra
    - "metriche": ore di luce, ore di comunicazione, tempo al tramonto, ecc.

Convenzioni (da condividere con il team!):
    - Tempi in UTC, formato ISO 8601 (es. "2026-11-13T18:00:00Z")
    - Angoli in GRADI
    - Azimut misurato da NORD (0) verso EST (90), SUD (180), OVEST (270)
    - Latitudine positiva a nord, longitudine positiva a est
    - Orizzonte: lista di 360 elevazioni (gradi), una per ogni grado di
      azimut 0..359. Se None si usa l'orizzonte "liscio" (sfera, 0 gradi).
      La coppia Terreno fornira' l'orizzonte reale da LOLA.

Nota: Sole e Terra sono considerati "visibili" quando il loro CENTRO e' sopra
l'orizzonte (raggio apparente Sole ~0.27 gradi, Terra ~0.95 gradi).
Per la POTENZA invece si usa la frazione del disco solare visibile sopra il
terreno (il Sole che "sfiora" una cresta da' solo una parte dell'energia).

Effetti ambientali inclusi (valori e riferimenti in FONTI.txt):
    - Margine di comunicazione: la Terra e' utile per il link diretto solo se
      e' almeno `margine_com_deg` sopra il terreno (diffrazione sulle creste /
      zona di Fresnel, multipath sulla regolite a basso angolo).
    - Polvere sui pannelli: perdita iniziale da atterraggio (scenario) +
      degrado graduale annuo (dati Apollo Dust Detector).
    - Terminatore / polvere elettrostatica: ore con Sole radente e numero di
      transizioni luce-buio (indicatore qualitativo di rischio).
    - Buio continuo piu' lungo: periodo da sopravvivere con le batterie
      (regolite isolante -> temperature molto basse all'ombra).
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from skyfield.api import Loader, PlanetaryConstants

# I file delle effemeridi vengono salvati in 1-Ephemerids/dati_effemeridi
# (sono esclusi da Git tramite .gitignore e si riscaricano da soli).
_CARTELLA_DATI = Path(__file__).resolve().parent / "dati_effemeridi"
_CARTELLA_DATI.mkdir(exist_ok=True)
_load = Loader(str(_CARTELLA_DATI), verbose=True)

_cache = {}

# Costanti fisiche (vedi FONTI.txt)
RAGGIO_SOLE_KM = 695_700.0     # raggio solare nominale IAU 2015

# Valori di default degli effetti ambientali (vedi FONTI.txt)
MARGINE_COM_DEG = 0.5                 # Terra utile se >= 0.5 gradi sopra il terreno
PERDITA_INIZIALE_POLVERE_PCT = 0.0    # scenario "what-if" (nessun dato affidabile)
DEGRADO_POLVERE_ANNUO_PCT = 1.3       # Apollo DDE: 4-10% in 2000 giorni -> 0.7-1.8 %/anno
SOGLIA_SOLE_RADENTE_DEG = 5.0         # Sole "radente" se meno di 5 gradi sopra il terreno


def _risorse():
    """Carica effemeridi e orientamento lunare una sola volta (cache)."""
    if not _cache:
        ts = _load.timescale()
        eph = _load('de421.bsp')
        pc = PlanetaryConstants()
        pc.read_text(_load('moon_080317.tf'))                 # Lunear reference frame
        pc.read_text(_load('pck00008.tpc'))                   # Planetary Constants Kernel (PCK) con orientamento della Luna
        pc.read_binary(_load('moon_pa_de421_1900-2050.bpc'))  # Lunar orientation data (PA = principal axes) per il periodo 1900-2050
        frame = pc.build_frame_named('MOON_ME_DE421')         # Build the lunar reference frame using the DE421 ephemeris - coerent with LOLA data
        _cache.update(ts=ts, eph=eph, pc=pc, frame=frame)
    return _cache


def _maschera(orizzonte, azimut):
    """Elevazione dell'orizzonte (gradi) nelle direzioni 'azimut'."""
    if orizzonte is None:
        return np.zeros_like(azimut)                          # If there is no horizon data, assume a flat horizon at 0 degrees elevation           
    oriz = np.asarray(orizzonte, dtype=float)                 # oriz is an array of 360 elevation values (one for each azimuth degree)      
    gradi = np.linspace(0, 360, len(oriz), endpoint=False)    # Create an array of azimuth values corresponding to the elevation values in 'oriz'
    return np.interp(azimut, gradi, oriz, period=360)         


def _trova_eventi(tempi, margine, nome_sorge, nome_tramonta):
    """Trova gli istanti in cui 'margine' (elevazione - orizzonte) cambia segno.
    Usa un'interpolazione lineare tra due campioni consecutivi: con un passo
    di 10 minuti l'errore e' di pochi minuti, ampiamente sufficiente.
    """
    eventi = []
    for i in range(len(margine) - 1):                      # Loop through the 'margine' array to find sign changes between consecutive samples
        a, b = margine[i], margine[i + 1]
        if (a <= 0 < b) or (a > 0 >= b):
            frazione = a / (a - b)                         # interpolate to find the fraction of the interval where the sign change occurs
            t = tempi[i] + (tempi[i + 1] - tempi[i]) * frazione
            tipo = nome_sorge if b > a else nome_tramonta
            eventi.append({"tipo": tipo, "tempo": t})
    return eventi


def _ore(maschera_bool, passo_min):
# Compute the total number of hours where the boolean mask is True, given the time step in minutes                      
    return float(np.sum(maschera_bool) * passo_min / 60)


def _durata_testo(td):
    """timedelta -> '4g 7h 12m'."""
    if td is None:
        return None
    minuti = int(td.total_seconds() // 60)
    g, resto = divmod(minuti, 24 * 60)
    h, m = divmod(resto, 60)
    return f"{g}g {h}h {m}m"


def _iso(t):
    """datetime -> ISO 8601 (UTC) string."""
    return t.strftime('%Y-%m-%dT%H:%M:%SZ')               # For JAVA!!!


def _frazione_disco(u):
    """Frazione visibile di un disco tagliato da una linea orizzontale.

    u = (elevazione del centro - orizzonte) / raggio apparente del disco.
    u <= -1 -> disco tutto nascosto (0); u >= 1 -> tutto visibile (1);
    u = 0 -> meta' disco. Formula del segmento circolare; il profilo del
    terreno e' considerato localmente orizzontale sulla larghezza del disco.
    """
    u = np.clip(u, -1.0, 1.0)
    return 1.0 - (np.arccos(u) - u * np.sqrt(1.0 - u * u)) / np.pi


def _sequenza_piu_lunga(maschera_bool, passo_min):
    """Durata (ore) della sequenza consecutiva di True piu' lunga.

    Restituisce anche True se la sequenza tocca l'inizio o la fine del periodo
    (in quel caso la durata reale potrebbe essere maggiore).
    """
    migliore, corrente, inizio_corrente, tocca_bordo = 0, 0, 0, False
    for i, v in enumerate(maschera_bool):
        if v:
            if corrente == 0:
                inizio_corrente = i
            corrente += 1
            if corrente > migliore:
                migliore = corrente
                tocca_bordo = inizio_corrente == 0 or i == len(maschera_bool) - 1
        else:
            corrente = 0
    ore = migliore * passo_min / 60
    return round(float(ore), 1), bool(tocca_bordo and migliore > 0)


def calcola_posizioni(lat, lon, inizio, giorni=30, passo_min=10):
    """Elevazione e azimut di Sole e Terra visti dal sito, nel tempo.

    inizio: datetime (UTC). Restituisce (tempi, sole_alt, sole_az, terra_alt, terra_az).
    """
    tempi, s_alt, s_az, _, e_alt, e_az, _ = _calcola_completo(lat, lon, inizio, giorni, passo_min)
    return tempi, s_alt, s_az, e_alt, e_az


def _calcola_completo(lat, lon, inizio, giorni=30, passo_min=10):
    """Come calcola_posizioni, ma restituisce anche le distanze in km.

    Restituisce (tempi, sole_alt, sole_az, sole_dist_km, terra_alt, terra_az, terra_dist_km).
    """
    r = _risorse()
    ts, eph, pc, frame = r['ts'], r['eph'], r['pc'], r['frame']

    if inizio.tzinfo is None:
        inizio = inizio.replace(tzinfo=timezone.utc)
    n = int(giorni * 24 * 60 / passo_min) + 1
    tempi = [inizio + timedelta(minutes=passo_min * k) for k in range(n)]   # Create a list of datetime objects representing the time steps for the analysis
    t = ts.from_datetimes(tempi)               # Convert the list of datetime objects to Skyfield Time objects for ephemeris calculations (vectorial form)

    sito = eph['moon'] + pc.build_latlon_degrees(frame, lat, lon)     # Create a Skyfield position object for the lunar site using the Moon's ephemeris and the provided latitude and longitude
    osservatore = sito.at(t)                                          # Create a Skyfield observer object at the lunar site for the specified times

    risultati = []
    for corpo in (eph['sun'], eph['earth']):                          # Loop through the Sun and Earth to calculate their apparent positions as seen from the lunar site
        alt, az, dist = osservatore.observe(corpo).apparent().altaz() # Compute the apparent altitude and azimuth of the celestial body as seen from the lunar site
        risultati += [alt.degrees, az.degrees, dist.km]               # Altitude, azimuth (degrees) and distance (km) for both the Sun and Earth

    return (tempi, *risultati)  # tempi, sole_alt, sole_az, sole_dist, terra_alt, terra_az, terra_dist


def valuta_sito(lat, lon, inizio, giorni=30, passo_min=10, orizzonte=None,
                margine_com_deg=MARGINE_COM_DEG,
                perdita_iniziale_polvere_pct=PERDITA_INIZIALE_POLVERE_PCT,
                degrado_polvere_annuo_pct=DEGRADO_POLVERE_ANNUO_PCT,
                soglia_sole_radente_deg=SOGLIA_SOLE_RADENTE_DEG):
    """Valuta un sito lunare su un intervallo di tempo.

    Parametri
    ---------
    lat, lon   : coordinate del sito in gradi (lon positiva a est)
    inizio     : datetime UTC di inizio (es. data di atterraggio)
    giorni     : durata dell'analisi (30 = circa una giornata lunare)
    passo_min  : passo temporale in minuti
    orizzonte  : lista di 360 elevazioni dell'orizzonte (gradi) oppure None
    margine_com_deg              : gradi minimi della Terra sopra il terreno per un link utile
    perdita_iniziale_polvere_pct : perdita di potenza dovuta alla polvere dell'atterraggio (%)
    degrado_polvere_annuo_pct    : degrado graduale dei pannelli per polvere (% all'anno)
    soglia_sole_radente_deg      : sotto questa altezza sul terreno il Sole e' "radente"
    """
    tempi, s_alt, s_az, s_dist, e_alt, e_az, e_dist = _calcola_completo(lat, lon, inizio, giorni, passo_min)
    inizio = tempi[0]

    # Margine = quanto il corpo e' sopra l'orizzonte locale (terreno incluso)
    s_marg = s_alt - _maschera(orizzonte, s_az)                 # Compute the margin of the Sun's elevation above the local horizon (including terrain) by subtracting the interpolated horizon elevation from the Sun's altitude
    e_marg = e_alt - _maschera(orizzonte, e_az)                 # Compute the margin of the Earth's elevation above the local horizon (including terrain) by subtracting the interpolated horizon elevation from the Earth's altitude
    sole_su = s_marg > 0
    terra_su = e_marg > 0

    # --- Potenza: frazione di disco solare visibile sopra il terreno
    raggio_sole_deg = np.degrees(np.arcsin(RAGGIO_SOLE_KM / s_dist))
    frazione_sole = _frazione_disco(s_marg / raggio_sole_deg)

    # --- Polvere sui pannelli: perdita iniziale + degrado lineare nel tempo
    giorni_trascorsi = np.array([(t - tempi[0]).total_seconds() / 86400 for t in tempi])
    fattore_polvere = ((1 - perdita_iniziale_polvere_pct / 100)
                       * np.clip(1 - degrado_polvere_annuo_pct / 100 * giorni_trascorsi / 365.25, 0, 1))
    # Potenza relativa di un pannello che punta sempre il Sole (1 = disco pieno, pannelli puliti)
    potenza = frazione_sole * fattore_polvere

    # --- Comunicazione diretta con la Terra: serve un margine sopra il terreno
    comunicazione = e_marg > margine_com_deg

    # --- Terminatore / polvere elettrostatica: Sole visibile ma radente
    sole_radente = sole_su & (s_marg < soglia_sole_radente_deg)
    buio_max, buio_bordo = _sequenza_piu_lunga(~sole_su, passo_min)
    no_com_max, no_com_bordo = _sequenza_piu_lunga(~comunicazione, passo_min)

    eventi = (_trova_eventi(tempi, s_marg, 'alba', 'tramonto')
              + _trova_eventi(tempi, e_marg, 'sorgere_terra', 'tramonto_terra'))   # Find the events of sunrise/sunset for the Sun and rise/set for the Earth by checking where the margin changes sign (crosses zero) and store them in a list of events
    eventi.sort(key=lambda e: e['tempo'])   # Sort the list of events by their occurrence time in ascending order

    def prossimo(tipo):
        '''
        Return the timedelta until the next event of the specified type (e.g., 'alba', 'tramonto', 'sorgere_terra', 
        'tramonto_terra') from the start time. If no such event occurs within the analyzed period, return None.'''
        for e in eventi:
            if e['tipo'] == tipo:
                return e['tempo'] - inizio
        return None

    # Tempo al prossimo tramonto del Sole e della Terra (solo se ora sono visibili).
    # Se non tramontano entro il periodo analizzato, il testo lo dice esplicitamente.
    oltre = f"oltre {giorni}g (nessun tramonto nel periodo)"
    al_tramonto = prossimo('tramonto') if sole_su[0] else None
    al_tramonto_terra = prossimo('tramonto_terra') if terra_su[0] else None

    def testo_tramonto(visibile, td):
        '''Return a string describing the time until sunset for the Sun or Earth, or indicate if it is already below the horizon.'''
        if not visibile:
            return "gia' sotto l'orizzonte"
        return _durata_testo(td) if td is not None else oltre

    def testo_sorgere(visibile, td):
        '''Return a string describing the time until sunrise for the Sun or Earth, or indicate if it is already above the horizon.'''
        if visibile:
            return None
        return _durata_testo(td) if td is not None else f"oltre {giorni}g (non sorge nel periodo)"
    
    # Main matrics dictionaty - for json output
    metriche = {
        "sole_visibile_ora": bool(sole_su[0]),
        "terra_visibile_ora": bool(terra_su[0]),
        "elevazione_sole_ora": round(float(s_alt[0]), 2),
        "elevazione_terra_ora": round(float(e_alt[0]), 2),
        "tempo_al_tramonto": testo_tramonto(sole_su[0], al_tramonto),
        "ore_al_tramonto": None if al_tramonto is None else round(al_tramonto.total_seconds() / 3600, 1),
        "tempo_al_tramonto_terra": testo_tramonto(terra_su[0], al_tramonto_terra),
        "ore_al_tramonto_terra": None if al_tramonto_terra is None else round(al_tramonto_terra.total_seconds() / 3600, 1),
        "tempo_alla_prossima_alba": testo_sorgere(sole_su[0], prossimo('alba')),
        "tempo_al_sorgere_terra": testo_sorgere(terra_su[0], prossimo('sorgere_terra')),
        "ore_di_luce": round(_ore(sole_su, passo_min), 1),
        "ore_terra_visibile": round(_ore(terra_su, passo_min), 1),
        "ore_luce_e_comunicazione": round(_ore(sole_su & terra_su, passo_min), 1),
        "percentuale_luce": round(100 * float(np.mean(sole_su)), 1),
        "percentuale_terra_visibile": round(100 * float(np.mean(terra_su)), 1),
        "elevazione_sole_max": round(float(s_alt.max()), 2),
        "elevazione_terra_min": round(float(e_alt.min()), 2),
        "elevazione_terra_max": round(float(e_alt.max()), 2),

        # Energia (frazione di disco solare + polvere)
        "potenza_relativa_ora_pct": round(100 * float(potenza[0]), 1),
        "energia_relativa_media_pct": round(100 * float(np.mean(potenza)), 1),
        "ore_equivalenti_piena_potenza": round(float(np.sum(potenza) * passo_min / 60), 1),
        "ore_sole_parzialmente_coperto": round(_ore((frazione_sole > 0) & (frazione_sole < 1), passo_min), 1),
        "fattore_polvere_fine_periodo_pct": round(100 * float(fattore_polvere[-1]), 2),

        # Comunicazione utile (con margine sopra il terreno)
        "ore_comunicazione_utile": round(_ore(comunicazione, passo_min), 1),
        "percentuale_comunicazione_utile": round(100 * float(np.mean(comunicazione)), 1),
        "ore_luce_e_comunicazione_utile": round(_ore(sole_su & comunicazione, passo_min), 1),
        "senza_comunicazione_piu_lungo_ore": no_com_max,
        "senza_comunicazione_tocca_bordo_periodo": no_com_bordo,

        # Sopravvivenza al buio (regolite isolante -> freddo estremo)
        "buio_continuo_piu_lungo_ore": buio_max,
        "buio_tocca_bordo_periodo": buio_bordo,

        # Terminatore / polvere elettrostatica (indicatore qualitativo)
        "ore_sole_radente": round(_ore(sole_radente, passo_min), 1),
        "percentuale_luce_radente": (round(100 * float(np.sum(sole_radente)) / float(np.sum(sole_su)), 1)
                                     if np.any(sole_su) else None),
        "transizioni_luce_buio": sum(1 for e in eventi if e['tipo'] in ('alba', 'tramonto')),
    }
    metriche["rischio_polvere_elettrostatica"] = _rischio_polvere({**metriche, "_giorni": giorni})

    return {
        "sito": {"lat": lat, "lon": lon},
        "periodo": {"inizio": _iso(inizio), "giorni": giorni, "passo_min": passo_min},
        "orizzonte_terreno": orizzonte is not None,
        "ipotesi": {
            "margine_com_deg": margine_com_deg,
            "perdita_iniziale_polvere_pct": perdita_iniziale_polvere_pct,
            "degrado_polvere_annuo_pct": degrado_polvere_annuo_pct,
            "soglia_sole_radente_deg": soglia_sole_radente_deg,
            "pannello": "punta sempre il Sole; potenza proporzionale alla frazione di disco visibile",
            "fonti": "1-Ephemerids/FONTI.txt",
        },
        "metriche": metriche,
        "eventi": [{"tipo": e['tipo'], "tempo": _iso(e['tempo'])} for e in eventi],
        "serie": {
            "tempo": [_iso(t) for t in tempi],
            "sole_elevazione": np.round(s_alt, 3).tolist(),
            "sole_azimut": np.round(s_az, 3).tolist(),
            "terra_elevazione": np.round(e_alt, 3).tolist(),
            "terra_azimut": np.round(e_az, 3).tolist(),
            "sole_frazione_disco": np.round(frazione_sole, 4).tolist(),
            "potenza_relativa": np.round(potenza, 4).tolist(),
            "comunicazione_utile": comunicazione.tolist(),
        },
    }


def _rischio_polvere(m):
    """Indicatore QUALITATIVO del rischio di polvere elettrostatica.

    La polvere e' piu' mobile vicino al terminatore (Sole radente, forti
    differenze di carica tra zone illuminate e in ombra). Le soglie sono
    indicative, scelte dal team e non prese dalla letteratura: vedi FONTI.txt.
    """
    perc = m.get("percentuale_luce_radente")
    if perc is None:
        return "n/d (nessuna luce nel periodo)"
    # Una giornata lunare "normale" ha 2 transizioni ogni ~30 giorni: molte di
    # piu' significano Sole che entra ed esce dalle creste (terminatore mobile).
    transizioni_30g = m["transizioni_luce_buio"] * 30 / m["_giorni"]
    if perc >= 60 or transizioni_30g >= 6:
        return "alto"
    if perc >= 20 or transizioni_30g >= 4:
        return "medio"
    return "basso"


def grafico(risultato, titolo="", file=None):
    """Grafico di controllo: elevazione di Sole e Terra nel tempo."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    s = risultato['serie']
    tempi = [datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ') for t in s['tempo']]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tempi, s['sole_elevazione'], color='orange', label='Sole')
    ax.plot(tempi, s['terra_elevazione'], color='royalblue', label='Terra')
    ax.axhline(0, color='gray', lw=1, ls='--', label='Orizzonte (sfera liscia)')
    if 'potenza_relativa' in s:
        ax2 = ax.twinx()
        ax2.fill_between(tempi, [100 * p for p in s['potenza_relativa']], color='gold', alpha=0.15)
        ax2.set_ylabel('Potenza relativa [%] (area gialla)')
        ax2.set_ylim(0, 105)
    ax.set_ylabel('Elevazione [gradi]')
    ax.set_title(titolo)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    if file:
        fig.savefig(file, dpi=120)
    else:
        plt.show()
    plt.close(fig)


def _stampa(nome, r):
    m = r['metriche']
    print(f"\n=== {nome} ===")
    print(f"Sole ora: {m['elevazione_sole_ora']} gradi | Terra ora: {m['elevazione_terra_ora']} gradi")
    print(f"Tempo al tramonto del Sole:   {m['tempo_al_tramonto']}")
    print(f"Tempo al tramonto della Terra: {m['tempo_al_tramonto_terra']}")
    if m['tempo_alla_prossima_alba']:
        print(f"Prossima alba tra: {m['tempo_alla_prossima_alba']}")
    if m['tempo_al_sorgere_terra']:
        print(f"Prossimo sorgere della Terra tra: {m['tempo_al_sorgere_terra']}")
    print(f"Luce: {m['ore_di_luce']} h ({m['percentuale_luce']}%) | "
          f"Terra visibile: {m['ore_terra_visibile']} h ({m['percentuale_terra_visibile']}%) | "
          f"Entrambi: {m['ore_luce_e_comunicazione']} h")
    print(f"Elevazione max Sole: {m['elevazione_sole_max']} gradi | "
          f"Terra tra {m['elevazione_terra_min']} e {m['elevazione_terra_max']} gradi")
    print(f"Energia media: {m['energia_relativa_media_pct']}% | "
          f"Comunicazione utile (margine {r['ipotesi']['margine_com_deg']} gradi): {m['percentuale_comunicazione_utile']}% | "
          f"Luce + comunicazione utile: {m['ore_luce_e_comunicazione_utile']} h")
    print(f"Buio continuo piu' lungo: {m['buio_continuo_piu_lungo_ore']} h | "
          f"Senza comunicazione piu' lungo: {m['senza_comunicazione_piu_lungo_ore']} h")
    print(f"Sole radente: {m['ore_sole_radente']} h ({m['percentuale_luce_radente']}% della luce) | "
          f"Transizioni luce/buio: {m['transizioni_luce_buio']} | Rischio polvere: {m['rischio_polvere_elettrostatica']}")
    print("Primi eventi:")
    for e in r['eventi'][:6]:
        print(f"  {e['tempo']}  {e['tipo']}")


if __name__ == '__main__':
    # Esempio 1: Apollo 11, dall'allunaggio per 30 giorni
    apollo = valuta_sito(0.674, 23.473, datetime(1969, 7, 20, 20, 17, tzinfo=timezone.utc))
    _stampa("Apollo 11 (0.674 N, 23.473 E) dal 20/07/1969", apollo)

    # Esempio 2: vicino al polo sud, stessa durata, data recente
    polo = valuta_sito(-89.5, 0.0, datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc))
    _stampa("Polo sud (89.5 S, 0 E) dal 14/11/2026", polo)

    cartella = Path(__file__).resolve().parent / "output"
    cartella.mkdir(exist_ok=True)
    grafico(apollo, "Apollo 11 - elevazione di Sole e Terra", cartella / "grafico_apollo11.png")
    grafico(polo, "Polo sud (89.5 S) - elevazione di Sole e Terra", cartella / "grafico_polo_sud.png")
    print("\nGrafici salvati in:", cartella)
