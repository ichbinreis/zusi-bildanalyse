# -*- coding: utf-8 -*-
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
    SCRIPT_DIR = sys._MEIPASS  # type: ignore
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATEN_DIR = os.path.join(SCRIPT_DIR, "Daten")
CSV_PFAD = os.path.join(DATEN_DIR, "Objektdatenbank.csv")
BILDER_DIR = os.path.join(DATEN_DIR, "Bilder")
AUSGABE_DIR = os.path.join(SCRIPT_DIR, "ausgabe_bilder")
EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "alle_bilder_embeddings.npy")
INDEX_PATH = os.path.join(SCRIPT_DIR, "alle_bilder_index.csv")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "zusi_config.json")

TOP_N_IMAGE = 50  # endgültige Top-Kandidaten, die im HTML landen
VORFILTER_TOP_N = 200  # Standard-Modus: so viele Kandidaten kommen durch den Vorfilter

device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP
model_path = os.path.join(SCRIPT_DIR, "clip_patch", "open_clip_pytorch_model.bin")
model, _, image_preprocess = open_clip.create_model_and_transforms("ViT-B-32")
ckpt = torch.load(model_path, map_location="cpu")
state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()


def lade_bild(pfad):
    try:
        img = Image.open(pfad).convert("RGB")
        return image_preprocess(img).unsqueeze(0).to(device)
    except Exception:
        return None


def bilde_embedding(img_tensor):
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()


def zusi_verzeichnis_laden():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f).get("zusi_verzeichnis", "")
    return ""


def zusi_verzeichnis_speichern(pfad):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({"zusi_verzeichnis": pfad}, f)

def resolve_verzeichnis_path(verzeichnis, zusi_pfad):
    v = str(verzeichnis).strip()
    if not v or v == "NICHT GEFUNDEN":
        return ""
    v = v.replace("/", "\\")
    # absoluter Pfad?
    if os.path.isabs(v):
        lower = v.lower()
        marker = "_zusidata"
        i = lower.find(marker)
        if i != -1:
            rel = v[i + len(marker):].lstrip("\\/")
            return os.path.join(zusi_pfad, rel)
        return v
    # relativer Pfad
    return os.path.join(zusi_pfad, v.lstrip("\\/"))




# Fortschritt & ETA
ergebnisse_counter = 0           # Fertige Eingabebilder
gesamt_ergebnisse = 1            # Anzahl Eingabebilder
start_zeit = time.time()
current_image_start = time.time()
durations = []                   # Dauer je fertigem Eingabebild (Sekunden)

total_images = 1


def format_sec(s):
    s = int(max(0, s))
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"



def fortschritt_thread():
    global durations, current_image_start, total_images
    while ergebnisse_counter < gesamt_ergebnisse:
        time.sleep(1)
        done = ergebnisse_counter
        total = max(1, gesamt_ergebnisse)
        # Fortschritt in % (fertige Bilder / Gesamt)
        fortschritt.set(int(done / total * 100))

        elapsed_curr = time.time() - current_image_start

        # Per-Image ETA
        if len(durations) == 0:
            # Erste Schätzung: aktuelles Bild total = elapsed * 2 (Heuristik), min 10s
            est_per_image = max(10, elapsed_curr * 2)
            eta_curr = int(max(0, est_per_image - elapsed_curr))
            # Gesamt-ETA = Rest dieses Bildes + Restbilder * est_per_image
            remaining_images_after_current = max(0, total_images - done - 1)
            eta_total = int(eta_curr + remaining_images_after_current * est_per_image)
        else:
            avg = sum(durations) / len(durations)
            eta_curr = int(max(0, avg - elapsed_curr))
            remaining_images_after_current = max(0, total_images - done - 1)
            eta_total = int(eta_curr + remaining_images_after_current * avg)

        eta_text_var.set(f"ETA gesamt: {format_sec(eta_total)} | aktuelles Bild: {format_sec(eta_curr)}")
        root.update_idletasks()


# GUI
root = tk.Tk()
root.title("Zusi Bildvergleich")
root.resizable(False, False)

eingabe_pfade = []

# Nur noch zwei Modi: Standard (ehemals fast) und Strong
modus_var = tk.StringVar(value="standard")  # standard, strong
zusi_pfad_var = tk.StringVar(value=zusi_verzeichnis_laden())
ergebnis_oeffnen_var = tk.BooleanVar(value=True)
ordner_oeffnen_var = tk.BooleanVar(value=False)

progress_text_var = tk.StringVar(value="Bereit")
eta_text_var = tk.StringVar(value="")
fortschritt = tk.IntVar(value=0)


def bilder_auswaehlen():
    files = filedialog.askopenfilenames(
        title="Bilder auswählen",
        filetypes=[("Bilder", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff")]
    )
    if files:
        eingabe_pfade.clear()
        eingabe_pfade.extend(files)
        progress_text_var.set(f"{len(eingabe_pfade)} Bild(er) ausgewählt")
        root.update_idletasks()


def analyse_ausfuehren():
    global ergebnisse_counter, gesamt_ergebnisse, start_zeit, current_image_start, durations, total_images

    if not eingabe_pfade:
        messagebox.showwarning("Hinweis", "Keine Eingabebilder ausgewählt.")
        return

    modus = modus_var.get()
    zusi_pfad = zusi_pfad_var.get().strip()
    if not zusi_pfad:
        messagebox.showwarning("Hinweis", "Bitte Zusi-Stammdatenverzeichnis wählen.")
        return
    zusi_verzeichnis_speichern(zusi_pfad)

    # CSV laden – Auto-Delimiter (Komma üblich)
    if not os.path.exists(CSV_PFAD):
        messagebox.showerror("Fehler", f"CSV nicht gefunden: {CSV_PFAD}")
        return
    try:
        df = pd.read_csv(CSV_PFAD, sep=",", encoding="utf-8", dtype=str).fillna("")
    except Exception:
        df = pd.read_csv(CSV_PFAD, sep=",", encoding="latin1", dtype=str).fillna("")

    # Spalten-Mapping auf deine CSV
    need = ["jpg_dateiname", "Beschreibung", "Tatsächlicher Name", "Link", "Verzeichnis", "Vermutung"]  # "ungeeignet" optional
    for c in ["jpg_dateiname", "Link", "Verzeichnis", "Vermutung"]:
        if c not in df.columns:
            messagebox.showerror("Fehler", f"Spalte fehlt in CSV: {c}")
            return

    # Name-Feld wählen (Tatsächlicher Name bevorzugt, sonst Beschreibung)
    name_col = "Tatsächlicher Name" if "Tatsächlicher Name" in df.columns else ("Beschreibung" if "Beschreibung" in df.columns else None)
    if name_col is None:
        messagebox.showerror("Fehler", "Spalte fehlt in CSV: 'Tatsächlicher Name' oder 'Beschreibung'")
        return

    # Index vorbereiten
    alle_bilder = []
    bildname2info = {}
    for _, row in df.iterrows():
        bild = row.get("jpg_dateiname", "").strip()
        if not bild:
            continue
        alle_bilder.append(bild)
        bildname2info[bild] = {
            "name": row.get(name_col, "").strip(),
            "link": row.get("Link", "").strip(),
            "verzeichnis": row.get("Verzeichnis", "").strip(),
            "vermutung": str(row.get("Vermutung", "")).strip(),
            "ungeeignet": str(row.get("ungeeignet", "")).strip(),
        }

    if not alle_bilder:
        messagebox.showerror("Fehler", "Keine Einträge in der CSV.")
        return

    # Vorfilter vorbereiten (nur Standard-Modus)
    if modus == "standard":
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(INDEX_PATH):
            messagebox.showerror("Fehler", "Standard-Modus benötigt vorberechnete Embeddings.")
            return
        alle_vecs = np.load(EMBEDDINGS_PATH)
        index_df = pd.read_csv(INDEX_PATH)
        if "Bildname" in index_df.columns:
            idx_col = "Bildname"
        elif "jpg_dateiname" in index_df.columns:
            idx_col = "jpg_dateiname"
        else:
            # Fallback: erste Spalte als Name verwenden
            idx_col = index_df.columns[0]
        bild2row = {str(bn): i for i, bn in enumerate(index_df[idx_col].astype(str).tolist())}
    else:
        alle_vecs = None
        bild2row = None

    # Ausgabeordner
    os.makedirs(AUSGABE_DIR, exist_ok=True)

    # Fortschritt pro Eingabebild
    ergebnisse_counter = 0
    gesamt_ergebnisse = len(eingabe_pfade)
    total_images = gesamt_ergebnisse
    durations = []
    start_zeit = time.time()
    threading.Thread(target=fortschritt_thread, daemon=True).start()

    for idx, eingabe_pfad in enumerate(eingabe_pfade, start=1):
        current_image_start = time.time()
        eingabe_datei = os.path.basename(eingabe_pfad)
        progress_text_var.set(f"Analysiere {eingabe_datei} ({idx}/{len(eingabe_pfade)})...")
        root.update_idletasks()

        img_t = lade_bild(eingabe_pfad)
        if img_t is None:
            messagebox.showwarning("Warnung", f"Konnte Bild nicht laden: {eingabe_pfad}")
            durations.append(time.time() - current_image_start)
            ergebnisse_counter += 1
            continue
        q = bilde_embedding(img_t)

        # Kandidatenliste ermitteln
        if modus == "standard":
            # Index-Zuordnung
            vec_index = [bild2row.get(str(b), None) for b in alle_bilder]
            ok_mask = [i for i, v in enumerate(vec_index) if v is not None]
            if not ok_mask:
                messagebox.showerror("Fehler", "Index stimmt nicht mit CSV überein.")
                return
            cand_vecs = alle_vecs[[vec_index[i] for i in ok_mask]]
            sims = cosine_similarity(q, cand_vecs)[0]
            top_idx = np.argsort(-sims)[:VORFILTER_TOP_N]
            kandidaten = [alle_bilder[ok_mask[i]] for i in top_idx]
        else:  # strong
            kandidaten = alle_bilder

        # Feinauswahl: TOP_N_IMAGE
        cand_paths = [os.path.join(BILDER_DIR, k) for k in kandidaten]
        cand_tensors = []
        cand_names_ok = []
        for pth in cand_paths:
            t = lade_bild(pth)
            if t is None:
                continue
            cand_tensors.append(t)
            cand_names_ok.append(os.path.basename(pth))
        if not cand_tensors:
            durations.append(time.time() - current_image_start)
            ergebnisse_counter += 1
            continue

        cand_batch = torch.cat(cand_tensors, dim=0)
        with torch.no_grad():
            emb_cand = model.encode_image(cand_batch.to(device))
            emb_cand = emb_cand / emb_cand.norm(dim=-1, keepdim=True)
            emb_cand = emb_cand.cpu().numpy()
        sims2 = cosine_similarity(q, emb_cand)[0]

        top_idx2 = np.argsort(-sims2)[:TOP_N_IMAGE]
        top = []
        for i in top_idx2:
            bname = cand_names_ok[i]
            info = bildname2info.get(bname, {})
            top.append((float(sims2[i]), bname, info.get("name", ""), info.get("link", ""),
                        info.get("verzeichnis", ""), info.get("vermutung", ""), info.get("ungeeignet", "")))

        # Ausgabeordner
        basisname = os.path.splitext(os.path.basename(eingabe_pfad))[0]
        ausgabe_pfad = os.path.join(AUSGABE_DIR, basisname)
        os.makedirs(ausgabe_pfad, exist_ok=True)

        # HTML schreiben
        # Eingabebild in Ausgabeordner kopieren
        try:
            shutil.copy2(eingabe_pfad, os.path.join(ausgabe_pfad, os.path.basename(eingabe_pfad)))
        except Exception:
            pass
        html_path = os.path.join(ausgabe_pfad, "index.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("""<!doctype html>
<html lang="de"><head><meta charset="utf-8">
<title>Ergebnis</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
.item { border: 1px solid #ddd; border-radius: 8px; padding: 10px; }
.thumb { width: 100%; height: auto; border-radius: 6px; }
.meta { font-size: 12px; color: #333; }

.warn-flag { position: relative; cursor: help; display: inline-block; }
.warn-flag .tooltip {
  visibility: hidden;
  opacity: 0;
  transition: opacity 0.2s;
  position: absolute;
  bottom: 125%;
  left: 0;
  max-width: 320px;
  background: #fffbea;
  color: #222;
  border: 1px solid #f2d024;
  border-radius: 6px;
  padding: 8px 10px;
  box-shadow: 0 6px 18px rgba(0,0,0,.15);
  z-index: 10;
}
.warn-flag:hover .tooltip { visibility: visible; opacity: 1; }
.warn-flag .tooltip::after {
  content: "";
  position: absolute;
  top: 100%;
  left: 12px;
  border-width: 6px;
  border-style: solid;
  border-color: #f2d024 transparent transparent transparent;
}

</style></head><body>
<h2>Eingabebild</h2>""" + f'<img src="{os.path.basename(eingabe_pfad)}" style="max-width:100%;height:auto;margin-bottom:20px">' + """<h2>Ähnlichste Bilder zu """ + basisname + """</h2>
<div class='grid'>""")
            for score, bildname, name, link, verzeichnis, vermutung, ungeeignet_val in top:
                # Warnhinweis je Objekt
                warn_html_parts = []
                if str(vermutung).strip().lower() == "vermutung":
                    tooltip_text = (
                        f'Hintergrund: Dieses Objekt heißt "{bildname}" auf der Objektdatenbank. '
                        'Der Pfad und/oder der Name des Zusi-Objekts im offiziellen Bestand ist nicht exakt der gleiche. '
                        'Mit einer Ähnlichkeitssuche wurde dieses und weitere Objekte automatisch zugeordnet. '
                        'Das kann bei manchen Objekten falsch sein.'
                    )
                    warn_html_parts.append(
                        "<span class='warn-flag'>⚠️"
                        f"<span class='tooltip'>{tooltip_text}</span>"
                        "</span> "
                        "<span style='color:#b00020;font-weight:600;'>Hinweis: Die Daten zu diesem Objekt sind womöglich nicht korrekt (Dateiname und Verzeichnis können abweichen).</span>"
                    )
                if str(ungeeignet_val).strip().lower() == "ungeeignet":
                    warn_html_parts.append(
                        "<div style='color:#b00020;font-weight:600;margin-top:4px'>⚠️ Hinweis: Dieses Objekt ist vermutlich ungeeignet für den Geländeformer.</div>"
                    )
                warn_html = "".join(warn_html_parts)

                src_img = os.path.join(BILDER_DIR, bildname)
                dst_img = os.path.join(ausgabe_pfad, bildname)
                try:
                    shutil.copy2(src_img, dst_img)
                except Exception:
                    continue

                link_html = f"<a href='{link}' target='_blank'>Link</a><br>" if link and link != "-" else ""
                verzeichnis_pfad = resolve_verzeichnis_path(verzeichnis, zusi_pfad)
                if verzeichnis_pfad:
                    verzeichnis_url = 'file:///' + verzeichnis_pfad.replace("\\", "/").replace(" ", "%20")
                    path_for_clip = verzeichnis_pfad.replace("\\", "/")
                    link_objekt = (
                        f"<a href='{verzeichnis_url}' target='_blank'>Zum Objektordner</a><br>"
                        f"<a href='#' onclick=\"navigator.clipboard.writeText('{path_for_clip}'); return false;\">📋 Pfad kopieren</a>"
                        f"{warn_html}<br>"
                    )
                    anzeige_verzeichnis = verzeichnis_pfad.replace("/", "\\")
                    anzeige_verzeichnis_html = f"<div style='font-size:10px; color:#555; word-break:break-all;'>{anzeige_verzeichnis}</div>"
                else:
                    link_objekt = ("<span title='Dieses Objekt scheint nicht im offiziellen Bestand zu liegen' style='color:gray'>Kein Objektordner</span>"
                                   f"{warn_html}<br>")
                    anzeige_verzeichnis_html = ""

                f.write(
                    f"<div class='item'>"
                    f"<b>Score: {score:.2f}</b><div class='meta'>{name}</div>"
                    f"{link_html}{link_objekt}{anzeige_verzeichnis_html}"
                    f"<img src='{bildname}' class='thumb'>"
                    f"</div>"
                )
            f.write("</div></body></html>")

        if ergebnis_oeffnen_var.get():
            try:
                webbrowser.open(f"file:///{html_path}")
            except Exception:
                pass

        if ordner_oeffnen_var.get():
            try:
                subprocess.Popen(["explorer", os.path.realpath(ausgabe_pfad)])
            except Exception:
                pass

        progress_text_var.set(f"Analysiere {eingabe_datei} ({idx}/{len(eingabe_pfade)})...")
        root.update_idletasks()

        # Fortschritt pro Bild
        durations.append(time.time() - current_image_start)
        ergebnisse_counter += 1

    if ergebnis_oeffnen_var.get():
        try:
            webbrowser.open(f"file:///{html_path}")
        except Exception:
            pass

    fortschritt.set(100)
    eta_text_var.set("Fertig")
    progress_text_var.set("Analyse abgeschlossen")


# Layout
frm = tk.Frame(root)
frm.pack(padx=12, pady=12)

tk.Button(frm, text="Bilder auswählen", command=bilder_auswaehlen).grid(row=0, column=0, sticky="w")
tk.Label(frm, textvariable=progress_text_var).grid(row=0, column=1, sticky="w", padx=8)

tk.Label(root, text="Modus:").pack(anchor="w", padx=12)
for label, value in [("Standard", "standard"), ("Strong", "strong")]:
    tk.Radiobutton(root, text=label, variable=modus_var, value=value).pack(anchor="w", padx=18)

tk.Label(root, text="Zusi-Stammdatenverzeichnis (Ordner _ZusiData wählen):").pack(anchor="w", padx=12, pady=(8, 0))
tk.Entry(root, textvariable=zusi_pfad_var, width=60).pack(anchor="w", padx=12)
tk.Button(root, text="Pfad auswählen", command=lambda: zusi_pfad_var.set(filedialog.askdirectory())).pack(anchor="w", padx=12, pady=(4, 8))

tk.Checkbutton(root, text="Ergebnis sofort öffnen", variable=ergebnis_oeffnen_var).pack(anchor="w", padx=12)
tk.Checkbutton(root, text="Verzeichnis sofort öffnen", variable=ordner_oeffnen_var).pack(anchor="w", padx=12)

tk.Button(root, text="Analyse starten",
          command=lambda: threading.Thread(target=analyse_ausfuehren, daemon=True).start()).pack(pady=8, padx=12)

ttk.Progressbar(root, maximum=100, variable=fortschritt, length=320, mode="determinate").pack(anchor="w", padx=12)
tk.Label(root, textvariable=eta_text_var).pack(anchor="w", padx=12, pady=(4, 0))

root.mainloop()
