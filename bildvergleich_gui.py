# -*- coding: utf-8 -*-
# Bildvergleich GUI mit Verzeichniswahl, Fortschrittsanzeige, ETA und Fehlerbehandlung

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import webbrowser
import time
import json
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

import sys

if getattr(sys, 'frozen', False):
    SCRIPT_DIR = sys._MEIPASS
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATEN_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PFAD = os.path.join(DATEN_DIR, "Objektdatenbank.csv")
BILDER_DIR = os.path.join(DATEN_DIR, "Bilder")
AUSGABE_DIR = os.path.join(DATEN_DIR, "ausgabe_bilder")
EMBEDDINGS_PATH = os.path.join(DATEN_DIR, "alle_bilder_embeddings.npy")
INDEX_PATH = os.path.join(DATEN_DIR, "alle_bilder_index.csv")
CONFIG_PATH = os.path.join(DATEN_DIR, "zusi_config.json")

TOP_N_IMAGE = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = os.path.join(SCRIPT_DIR, "clip_patch", "open_clip_pytorch_model.bin")
model, _, image_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained=model_path
)
model.to(device)
model.eval()

def embed_image(pfad):
    try:
        image = image_preprocess(Image.open(pfad).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            return model.encode_image(image).cpu().numpy()
    except Exception as e:
        print(f"[WARN] Bild konnte nicht verarbeitet werden: {pfad} — {e}")
        return None

def zusi_verzeichnis_laden():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f).get("zusi_verzeichnis", "")
    return "C:/Program Files/Zusi3/_ZusiData"

def zusi_verzeichnis_speichern(pfad):
    with open(CONFIG_PATH, 'w') as f:
        json.dump({"zusi_verzeichnis": pfad}, f)

# Fortschritt & ETA live anzeigen
ergebnisse_counter = 0
gesamt_ergebnisse = 1
start_zeit = time.time()

def fortschritt_thread():
    while ergebnisse_counter < gesamt_ergebnisse:
        time.sleep(1)
        if ergebnisse_counter == 0:
            eta_text_var.set("ETA: wird berechnet ...")
            continue
        vergangen = time.time() - start_zeit
        rest = gesamt_ergebnisse - ergebnisse_counter
        eta = int((vergangen / ergebnisse_counter) * rest)
        fortschritt.set(int(ergebnisse_counter / gesamt_ergebnisse * 100))
        eta_text_var.set(f"ETA: {eta} Sekunden verbleibend")
        root.update_idletasks()

def analyse_ausfuehren():
    modus = modus_var.get()
    if modus not in ["fast", "medium", "strong"]:
        messagebox.showerror("Fehler", "Bitte einen Modus auswählen.")
        return

    zusi_pfad = zusi_pfad_var.get().strip()
    if not zusi_pfad or not os.path.exists(zusi_pfad):
        messagebox.showerror("Fehler", "Bitte ein gültiges Zusi-Stammverzeichnis auswählen.")
        return

    zusi_verzeichnis_speichern(zusi_pfad)

    eingabepfade = filedialog.askopenfilenames(title="Eingabebilder wählen", filetypes=[["Bilder", "*.jpg *.jpeg *.png"]])
    if not eingabepfade:
        return

    df_meta = pd.read_csv(CSV_PFAD, encoding="utf-8")
    df_meta = df_meta[df_meta['jpg_dateiname'].notna()]
    df_meta.set_index("jpg_dateiname", inplace=True)
    alle_bilder = df_meta.index.tolist()

    if modus == "fast":
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(INDEX_PATH):
            messagebox.showerror("Fehler", "Fast-Modus benötigt vorberechnete Embeddings.")
            return
        alle_vecs = np.load(EMBEDDINGS_PATH)
        index_df = pd.read_csv(INDEX_PATH)
        vorfilter_top_n = 200
    elif modus == "medium":
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(INDEX_PATH):
            messagebox.showerror("Fehler", "Medium-Modus benötigt vorberechnete Embeddings.")
            return
        alle_vecs = np.load(EMBEDDINGS_PATH)
        index_df = pd.read_csv(INDEX_PATH)
        vorfilter_top_n = 1000
    else:
        kandidaten = alle_bilder

    if not os.path.exists(AUSGABE_DIR):
        os.makedirs(AUSGABE_DIR)

    global ergebnisse_counter, gesamt_ergebnisse, start_zeit
    ergebnisse_counter = 0
    gesamt_ergebnisse = len(eingabepfade) * TOP_N_IMAGE
    start_zeit = time.time()
    threading.Thread(target=fortschritt_thread, daemon=True).start()

    for idx, eingabe_pfad in enumerate(eingabepfade, start=1):
        eingabe_datei = os.path.basename(eingabe_pfad)
        ordnername = os.path.splitext(eingabe_datei)[0]
        ausgabe_pfad = os.path.join(AUSGABE_DIR, ordnername)

        eingabe_vec = embed_image(eingabe_pfad)
        if eingabe_vec is None:
            continue

        os.makedirs(ausgabe_pfad, exist_ok=True)
        shutil.copy2(eingabe_pfad, os.path.join(ausgabe_pfad, eingabe_datei))

        if modus in ["fast", "medium"]:
            scores = cosine_similarity([eingabe_vec[0]], alle_vecs)[0]
            top_idx = np.argsort(scores)[::-1][:vorfilter_top_n]
            kandidaten = index_df.iloc[top_idx]["bildname"].tolist()

        ergebnisse = []
        for kandidat in kandidaten:
            pfad = os.path.join(BILDER_DIR, kandidat)
            vec = embed_image(pfad)
            if vec is None:
                continue
            score = cosine_similarity(eingabe_vec, vec)[0][0]
            row = df_meta.loc[kandidat] if kandidat in df_meta.index else {}
            name = row.get("Tatsächlicher Name", "-")
            link = str(row.get("Link", "-")).strip()
            verzeichnis = str(row.get("Verzeichnis", "")).strip()
            vermutung = str(row.get("K", "")).strip()
            ergebnisse.append((score, kandidat, name, link, verzeichnis, vermutung))
            ergebnisse_counter += 1

        ergebnisse.sort(key=lambda x: x[0], reverse=True)
        top = ergebnisse[:TOP_N_IMAGE]

        html_path = os.path.join(ausgabe_pfad, "ergebnisse.html")
        os.makedirs(ausgabe_pfad, exist_ok=True)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"""<html><head><meta charset='utf-8'>
<title>Bildvergleich Ergebnisse</title>
<style>
body {{ font-family: sans-serif; padding: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
.item {{ border: 1px solid #ccc; padding: 10px; border-radius: 8px; background: #f9f9f9; }}
img.thumb {{ width: 100%; height: auto; border-radius: 4px; }}
img.header {{ max-width: 100%; max-height: 400px; display: block; margin: 0 auto; }}
</style></head><body>
<h2>Top {TOP_N_IMAGE} ähnlichste Bilder zu: {eingabe_datei}</h2>
<img src='{eingabe_datei}' class='header'><br><br><hr>
<div class='grid'>""")
            for score, bildname, name, link, verzeichnis, vermutung in top:
                zielbild = os.path.join(ausgabe_pfad, bildname)
                shutil.copy2(os.path.join(BILDER_DIR, bildname), zielbild)
                link_html = f"<a href='{link}' target='_blank'>Link</a><br>" if link and link != "-" else ""
                info = " <span title='ggf. falsches Verzeichnis – Dateiname weicht ab'>ℹ️</span>" if vermutung.lower() == "vermutung" else ""
                if verzeichnis and verzeichnis != "NICHT GEFUNDEN":
                    verzeichnis_pfad = os.path.join(zusi_pfad, verzeichnis.strip("/\\"))
                    verzeichnis_url = 'file:///' + verzeichnis_pfad.replace("\\", "/").replace(" ", "%20")
                    link_objekt = (
                        f"<a href='{verzeichnis_url}' target='_blank'>Zum Objektordner</a>{info}<br>"
                        f"<a href='#' onclick=\"navigator.clipboard.writeText('{verzeichnis_pfad.replace('\\', '/')}'); return false;\">📋 Pfad kopieren</a><br>"
                    )
                    anzeige_verzeichnis = verzeichnis_pfad.replace("/", "\\")
                    anzeige_verzeichnis_html = f"<div style='font-size:10px; color:#555; word-break:break-all;'>{anzeige_verzeichnis}</div>"

                else:
                    link_objekt = f"<span title='Dieses Objekt scheint nicht im offiziellen Bestand zu liegen' style='color:gray'>Kein Objektordner</span>{info}<br>"
                    anzeige_verzeichnis_html = ""
                f.write(f"<div class='item'><b>Score: {score:.2f}</b><br>{name}<br>{link_html}{link_objekt}{anzeige_verzeichnis_html}<img src='{bildname}' class='thumb'></div>")
            f.write("</div></body></html>")

        if ergebnis_oeffnen_var.get():
            webbrowser.open(f"file:///{html_path}")

        if ordner_oeffnen_var.get():
            subprocess.Popen(["explorer", os.path.realpath(ausgabe_pfad)])

        progress_text_var.set(f"Analysiere {eingabe_datei} ({idx}/{len(eingabepfade)})...")

    fortschritt.set(100)
    eta_text_var.set("Fertig!")
    progress_text_var.set(f"✅ Analyse abgeschlossen ({len(eingabepfade)} Bild(er))")

# GUI Setup
root = tk.Tk()
root.title("Zusi Bildvergleich")
root.geometry("480x360")
root.resizable(False, False)

modus_var = tk.StringVar(value="fast")
zusi_pfad_var = tk.StringVar(value=zusi_verzeichnis_laden())
progress_text_var = tk.StringVar()
eta_text_var = tk.StringVar()
fortschritt = tk.IntVar()
ergebnis_oeffnen_var = tk.BooleanVar(value=True)
ordner_oeffnen_var = tk.BooleanVar(value=False)

tk.Label(root, text="Verarbeitungsmodus wählen:").pack()
for m in [("Fast (~15s)", "fast"), ("Medium (~1min)", "medium"), ("Strong (~5min)", "strong")]:
    tk.Radiobutton(root, text=m[0], variable=modus_var, value=m[1]).pack()

tk.Label(root, text="Zusi-Stammdatenverzeichnis (Ordner _ZusiData wählen):").pack()
tk.Entry(root, textvariable=zusi_pfad_var, width=60).pack()
tk.Button(root, text="Pfad auswählen", command=lambda: zusi_pfad_var.set(filedialog.askdirectory())).pack()

tk.Checkbutton(root, text="Ergebnis sofort öffnen", variable=ergebnis_oeffnen_var).pack()
tk.Checkbutton(root, text="Verzeichnis sofort öffnen", variable=ordner_oeffnen_var).pack()
tk.Button(root, text="Bilder auswählen und Analyse starten", command=lambda: threading.Thread(target=analyse_ausfuehren).start()).pack()

tk.Label(root, textvariable=progress_text_var).pack()
tk.Label(root, textvariable=eta_text_var).pack()
ttk.Progressbar(root, maximum=100, variable=fortschritt, length=300, mode="determinate").pack()

root.mainloop()