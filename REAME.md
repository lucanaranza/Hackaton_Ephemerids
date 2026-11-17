# Hackaton_Ephemerids — Sole e Terra visti da un sito lunare

1) mappatura.py takes a custom path for the CSV celle data --> doesnt hold copy in ephemerids folder
2) runna subscript ephemerids.py per calcolo effemeridi 
3) prende dati JPL Horizon per effemeridi, scaricati nella cartella dati_effemeridi fino al 2050
4) calcola ostruzione terreno (da horizon data) rispetto a posizione elevation e azimuth SOLE Terra
6) calcola illuminazione, datalink in 2 modi
		* 1h temporal interval , STATIC --> doesnt save data at each time step, appends and calculates avg over 1 y
		* 1h temporal interval, DYNAMIC --> preallocates a tensor n_cells x n_cells x time_steps with zeros and compiles for each timestep
7) salva i dati in subfolders con nome sito
8) evoluzione = dynamic, mappa = static	 in cartella outvX	


COMANDO
Get-ChildItem "$topo\risultati_v2" -Directory | ForEach-Object { $csv = Join-Path $_.FullName "site_cells.csv"; if (Test-Path $csv) { $nome = $_.Name.Substring(0,1).ToUpper() + $_.Name.Substring(1); Write-Host "== $nome"; .\.venv\Scripts\python.exe mappatura.py --celle $csv --far $far --nome $nome --uscita "out_v2\$nome" --passo-tempo 60 --passo-evoluzione 60 } }


