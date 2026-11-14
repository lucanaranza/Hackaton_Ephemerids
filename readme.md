# Hackaton_Ephemerids — Sole e Terra visti da un sito lunare

Relative position for Earth and Sun, plus power / communication / dust metrics.

## File

| File | Contenuto |
|---|---|
| `effemeridi.py` | Motore: `valuta_sito(...)` calcola Sole/Terra nel tempo, eventi e metriche |
| `test_luna.py` | Validazione su Apollo 11 (confronto con i valori NASA ALSJ) |
| `FONTI.txt` | Tutte le fonti e le ipotesi dei valori numerici |
| `output/` | Grafici di controllo generati da `effemeridi.py` |
| `dati_effemeridi/` | File JPL/NAIF (inclusi nel repository, ~19 MB; se mancano si riscaricano da soli) |

## Uso rapido

```python
# eseguire dalla cartella principale di questo repository
from effemeridi import valuta_sito
from datetime import datetime

r = valuta_sito(-89.46, -137.49, datetime(2026, 11, 14),
                orizzonte=lista_360_valori,          # da gruppo Terreno (None = Luna liscia)
                margine_com_deg=0.5,                 # Terra utile se >= 0.5 gradi sopra il terreno
                perdita_iniziale_polvere_pct=0.0,    # scenario polvere dell'atterraggio
                degrado_polvere_annuo_pct=1.3)       # dati Apollo Dust Detector
print(r["metriche"])
```

Da terminale, nella cartella principale del repo:

```
python test_luna.py
python effemeridi.py
```

## Convenzioni

Tempi UTC (ISO 8601), angoli in gradi, azimut da Nord in senso orario,
longitudine positiva a Est, orizzonte = 360 valori (azimut 0..359).
