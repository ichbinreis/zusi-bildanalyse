# -*- coding: cp1252 -*-
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk

def _script_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS  # type: ignore
    return os.path.dirname(os.path.abspath(__file__))

SCRIPT_DIR = _script_dir()

# Schreibfähiger Cache für Modell-Downloads (wichtig für EXE)
def _openclip_cache_dir():
    base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    cache_dir = os.path.join(base, "ZusiBildvergleich", "openclip_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

OPENCLIP_CACHE_DIR = _openclip_cache_dir()
os.environ.setdefault("OPENCLIP_CACHE_DIR", OPENCLIP_CACHE_DIR)
os.environ.setdefault("TORCH_HOME", OPENCLIP_CACHE_DIR)

# ---------------- App ----------------
def start_gui():
    # GUI früh initialisieren, damit bei Ladefehlern ein Fenster/Fehlermeldung sichtbar ist
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    root = tk.Tk()
    root.title("Zusi Bildvergleich")
    root.resizable(False, False)

    # Imports + Abhängigkeiten prüfen
    try:
        import shutil, pandas as pd, numpy as np
        from PIL import Image
        import torch
        import open_clip
        from sklearn.metrics.pairwise import cosine_similarity
        import subprocess, webbrowser, time, json
    except ImportError as e:
        messagebox.showerror(
            "Fehlende Abhängigkeit",
            f"{e}\n\nBitte installiere:\n  pip install torch torchvision torchaudio\n  pip install open_clip_torch\n  pip install pillow pandas scikit-learn"
        )
        root.mainloop()
        return

    # rembg optional pruefen — echter Test-Import da rembg bei fehlendem
    # onnxruntime erst beim Benutzen (nicht beim Import) einen Fehler wirft
    rembg_remove = None
    rembg_verfuegbar = False
    try:
        import rembg as _rembg_mod
        from rembg import remove as rembg_remove
        # Echter Funktionstest: Session anlegen prueft ob onnxruntime da ist
        import io
        from PIL import Image as _PILTest
        _test_img = _PILTest.new("RGB", (4, 4), (128, 128, 128))
        _buf = io.BytesIO()
        _test_img.save(_buf, format="PNG")
        rembg_remove(_buf.getvalue())
        rembg_verfuegbar = True
    except Exception:
        rembg_remove = None
        rembg_verfuegbar = False

    # Pfade
    DATEN_DIR = os.path.join(SCRIPT_DIR, "Daten")
    CSV_PFAD = os.path.join(DATEN_DIR, "Objektdatenbank.csv")
    BILDER_DIR   = os.path.join(DATEN_DIR, "Bilder")
    BILDER_DIR_2 = os.path.join(DATEN_DIR, "Bilder_2")
    # Alle Bild-Ordner zusammen
    ALLE_BILDER_DIRS = [d for d in [BILDER_DIR, BILDER_DIR_2] if os.path.isdir(d)]
    AUSGABE_DIR = os.path.join(SCRIPT_DIR, "ausgabe_bilder")
    EMBEDDINGS_PATH = os.path.join(SCRIPT_DIR, "alle_bilder_embeddings.npy")
    INDEX_PATH = os.path.join(SCRIPT_DIR, "alle_bilder_index.csv")
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "zusi_config.json")

    TOP_N_IMAGE = 50
    VORFILTER_TOP_N = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- CLIP laden (CPU/GPU-kompatibel) ---
    try:
        model, _, image_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
            cache_dir=OPENCLIP_CACHE_DIR
        )
        model.to(device)
        model.eval()
    except Exception as e:
        messagebox.showerror(
            "Fehler beim Laden von CLIP",
            "Das CLIP-Modell konnte nicht geladen werden.\n\n"
            "Typische Ursachen:\n"
            "  • Beim ersten Start kein Internet (Gewichte müssen einmalig geladen werden)\n"
            "  • Firewall/Proxy blockiert den Download\n\n"
            "Falls der Fehler dauerhaft auftritt, können folgende Befehle in einer "
            "normalen Python-Umgebung helfen (nicht in der EXE, sondern in einer Konsole):\n"
            "  pip install torch torchvision torchaudio\n"
            "  pip install open_clip_torch\n"
            "  pip install pillow pandas scikit-learn\n\n"
            "Technische Meldung:\n"
            f"{e}"
        )
        root.mainloop()
        return

    # Helfer
    def lade_bild(pfad):
        try:
            img = Image.open(pfad).convert("RGB")
            return image_preprocess(img).unsqueeze(0).to(device)
        except Exception:
            return None

    def freistellen(pfad):
        """Entfernt den Hintergrund per rembg und gibt PIL-Image zurueck (RGBA).
        Bei Fehler oder rembg nicht verfuegbar: gibt None zurueck."""
        if not rembg_verfuegbar:
            return None
        try:
            with open(pfad, "rb") as f:
                inp = f.read()
            out = rembg_remove(inp)
            from PIL import Image as PILImage
            import io
            img = PILImage.open(io.BytesIO(out)).convert("RGBA")
            # Transparenten Hintergrund weiss faerben (CLIP arbeitet mit RGB)
            hintergrund = PILImage.new("RGB", img.size, (255, 255, 255))
            hintergrund.paste(img, mask=img.split()[3])
            return hintergrund
        except Exception:
            return None

    @torch.no_grad()
    def bilde_embedding_batch(img_tensors):
        batch = torch.cat(img_tensors, dim=0).to(device)
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()

    def zusi_verzeichnis_laden():
        # Config lesen, sonst Default vorbelegen
        default = r"C:\Program Files\Zusi3\_ZusiData"
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    val = json.load(f).get("zusi_verzeichnis", "").strip()
                    return val if val else default
            except Exception:
                return default
        return default

    def zusi_verzeichnis_speichern(pfad):
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump({"zusi_verzeichnis": pfad}, f)
        except Exception:
            pass

    def resolve_verzeichnis_path(verzeichnis, zusi_pfad):
        v = str(verzeichnis).strip()
        if not v or v == "NICHT GEFUNDEN":
            return ""
        v = v.replace("/", "\\")
        if len(v) >= 2 and v[1] == ":":
            lower = v.lower()
            marker = "_zusidata"
            i = lower.find(marker)
            if i != -1:
                rel = v[i + len(marker):].lstrip("\\/")
                return zusi_pfad.rstrip("\\/") + "\\" + rel.lstrip("\\/")
            return v
        return zusi_pfad.rstrip("\\/") + "\\" + v.lstrip("\\/")

    # Fortschritt & ETA
    ergebnisse_counter = 0
    gesamt_ergebnisse = 1
    start_zeit = 0.0
    current_image_start = 0.0
    durations = []
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
        nonlocal durations, current_image_start, total_images
        while ergebnisse_counter < gesamt_ergebnisse:
            time.sleep(1)
            done = ergebnisse_counter
            total = max(1, gesamt_ergebnisse)
            fortschritt.set(int(done / total * 100))
            elapsed_curr = time.time() - current_image_start if current_image_start else 0
            if len(durations) == 0 or elapsed_curr <= 0:
                est_per_image = max(10, elapsed_curr * 2 if elapsed_curr > 0 else 15)
                eta_curr = int(max(0, est_per_image - elapsed_curr))
                remaining_images_after_current = max(0, total_images - done - 1)
                eta_total = int(eta_curr + remaining_images_after_current * est_per_image)
            else:
                avg = sum(durations) / len(durations)
                eta_curr = int(max(0, avg - elapsed_curr))
                remaining_images_after_current = max(0, total_images - done - 1)
                eta_total = int(eta_curr + remaining_images_after_current * avg)
            eta_text_var.set(f"ETA gesamt: {format_sec(eta_total)} | aktuelles Bild: {format_sec(eta_curr)}")
            root.update_idletasks()

    # ===== Embeddings bauen =====
    def build_embeddings():
        if not os.path.exists(CSV_PFAD):
            messagebox.showerror("Fehler", f"CSV nicht gefunden: {CSV_PFAD}")
            return False
        try:
            df = pd.read_csv(CSV_PFAD, sep=",", encoding="utf-8", dtype=str).fillna("")
        except Exception:
            df = pd.read_csv(CSV_PFAD, sep=",", encoding="latin1", dtype=str).fillna("")
        if "jpg_dateiname" not in df.columns:
            messagebox.showerror("Fehler", "Spalte 'jpg_dateiname' fehlt.")
            return False
        dateinamen = [str(x).strip() for x in df["jpg_dateiname"].tolist() if str(x).strip()]
        if not dateinamen:
            messagebox.showerror("Fehler", "Keine Bilddateinamen in 'jpg_dateiname'.")
            return False

        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        imgs, kept_names, all_embs = [], [], []
        batch_size = 64 if torch.cuda.is_available() else 16

        progress_text_var.set("Erzeuge Embeddings...")
        root.update_idletasks()

        for name in dateinamen:
            # Suche Bild in allen Bilder-Ordnern
            p = None
            for bd in ALLE_BILDER_DIRS:
                kandidat = os.path.join(bd, name)
                if os.path.exists(kandidat):
                    p = kandidat
                    break
            if p is None:
                p = os.path.join(BILDER_DIR, name)  # Fallback (wird nicht existieren)
            t = lade_bild(p)
            if t is None:
                continue
            imgs.append(t)
            kept_names.append(name)
            if len(imgs) >= batch_size:
                progress_text_var.set(f"Erzeuge Embeddings ({len(kept_names)}/{len(dateinamen)})...")
                root.update_idletasks()
                embs = bilde_embedding_batch(imgs)
                all_embs.append(embs)
                imgs = []
        if imgs:
            embs = bilde_embedding_batch(imgs)
            all_embs.append(embs)

        if not kept_names or not all_embs:
            messagebox.showerror("Fehler", "Keine gültigen Bilder gefunden.")
            return False

        embs_total = np.concatenate(all_embs, axis=0)
        if embs_total.shape[0] != len(kept_names):
            m = min(embs_total.shape[0], len(kept_names))
            embs_total = embs_total[:m]
            kept_names = kept_names[:m]

        np.save(EMBEDDINGS_PATH, embs_total)
        pd.DataFrame({"Bildname": kept_names}).to_csv(INDEX_PATH, index=False, encoding="utf-8")
        messagebox.showinfo("Fertig", "Embeddings/Index erstellt.")
        return True

    # ---------- GUI-Elemente ----------
    eingabe_pfade = []
    hintergrund_entfernen_var = tk.BooleanVar(value=True)
    modus_var = tk.StringVar(value="standard")
    zusi_pfad_var = tk.StringVar(value=zusi_verzeichnis_laden())  # Default vorbelegt
    ergebnis_oeffnen_var = tk.BooleanVar(value=True)
    ordner_oeffnen_var = tk.BooleanVar(value=False)

    progress_text_var = tk.StringVar(value="Bereit")
    eta_text_var = tk.StringVar(value="")
    fortschritt = tk.IntVar(value=0)

    def bilder_auswaehlen():
        from tkinter import filedialog
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
        nonlocal ergebnisse_counter, gesamt_ergebnisse, start_zeit, current_image_start, durations, total_images

        if not eingabe_pfade:
            messagebox.showwarning("Hinweis", "Keine Eingabebilder ausgewählt.")
            return

        # Zusi-Pfad merken
        zusi_pfad = zusi_pfad_var.get().strip()
        if zusi_pfad:
            zusi_verzeichnis_speichern(zusi_pfad)

        # CSV prüfen
        if not os.path.exists(CSV_PFAD):
            messagebox.showerror("Fehler", f"CSV nicht gefunden: {CSV_PFAD}")
            return

        try:
            df = pd.read_csv(CSV_PFAD, sep=",", encoding="utf-8", dtype=str).fillna("")
        except Exception:
            df = pd.read_csv(CSV_PFAD, sep=",", encoding="latin1", dtype=str).fillna("")

        for c in ["jpg_dateiname", "Link", "Verzeichnis", "Vermutung"]:
            if c not in df.columns:
                messagebox.showerror("Fehler", f"Spalte fehlt in CSV: {c}")
                return

        name_col = "Tatsächlicher Name" if "Tatsächlicher Name" in df.columns else ("Beschreibung" if "Beschreibung" in df.columns else None)
        if name_col is None:
            messagebox.showerror("Fehler", "Spalte fehlt: 'Tatsächlicher Name' oder 'Beschreibung'")
            return

        alle_bilder, bildname2info = [], {}
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

        # Immer Standard-Modus mit Vorfilter (Embeddings)
        use_prefilter = False
        if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH)):
            ok = build_embeddings()
            if not ok:
                messagebox.showerror("Fehler", "Embeddings konnten nicht erstellt werden. Analyse abgebrochen.")
                return
        use_prefilter = True

        if use_prefilter:
            alle_vecs = np.load(EMBEDDINGS_PATH)
            index_df = pd.read_csv(INDEX_PATH)
            idx_col = "Bildname" if "Bildname" in index_df.columns else ("jpg_dateiname" if "jpg_dateiname" in index_df.columns else index_df.columns[0])
            bild2row = {str(bn): i for i, bn in enumerate(index_df[idx_col].astype(str).tolist())}
        else:
            alle_vecs = None
            bild2row = None

        os.makedirs(AUSGABE_DIR, exist_ok=True)

        # Fortschritt
        ergebnisse_counter = 0
        gesamt_ergebnisse = len(eingabe_pfade)
        total_images = gesamt_ergebnisse
        durations = []
        start_zeit = time.time()
        current_image_start = start_zeit
        threading.Thread(target=fortschritt_thread, daemon=True).start()

        for idx, eingabe_pfad in enumerate(eingabe_pfade, start=1):
            current_image_start = time.time()
            eingabe_datei = os.path.basename(eingabe_pfad)
            progress_text_var.set(f"Analysiere {eingabe_datei} ({idx}/{len(eingabe_pfade)})...")
            root.update_idletasks()

            # Hintergrund entfernen falls Checkbox aktiv
            if hintergrund_entfernen_var.get() and rembg_verfuegbar:
                progress_text_var.set(f"Freistellen {eingabe_datei} ({idx}/{len(eingabe_pfade)})...")
                root.update_idletasks()
                freigestellt = freistellen(eingabe_pfad)
                if freigestellt is not None:
                    img_t = image_preprocess(freigestellt).unsqueeze(0).to(device)
                else:
                    img_t = lade_bild(eingabe_pfad)
            else:
                img_t = lade_bild(eingabe_pfad)

            if img_t is None:
                messagebox.showwarning("Warnung", f"Konnte Bild nicht laden: {eingabe_pfad}")
                durations.append(time.time() - current_image_start)
                ergebnisse_counter += 1
                continue
            q = bilde_embedding_batch([img_t])[0:1]  # 1 x D

            if use_prefilter:
                vec_index = [bild2row.get(str(b), None) for b in alle_bilder]
                ok_idx = [i for i, v in enumerate(vec_index) if v is not None]
                if not ok_idx:
                    messagebox.showerror("Fehler", "Index stimmt nicht mit CSV überein.")
                    return
                cand_vecs = alle_vecs[[vec_index[i] for i in ok_idx]]
                sims = cosine_similarity(q, cand_vecs)[0]
                top_idx = np.argsort(-sims)[:VORFILTER_TOP_N]
                kandidaten = [alle_bilder[ok_idx[i]] for i in top_idx]
            else:
                kandidaten = alle_bilder

            # Pfad für jeden Kandidaten in beiden Ordnern suchen
            def finde_bild_pfad(name):
                for bd in ALLE_BILDER_DIRS:
                    p = os.path.join(bd, name)
                    if os.path.exists(p):
                        return p
                return os.path.join(BILDER_DIR, name)
            cand_paths = [finde_bild_pfad(k) for k in kandidaten]
            cand_tensors, cand_names_ok = [], []
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

            emb_cand = bilde_embedding_batch(cand_tensors)
            sims2 = cosine_similarity(q, emb_cand)[0]
            top_idx2 = np.argsort(-sims2)[:TOP_N_IMAGE]

            top = []
            for i2 in top_idx2:
                bname = cand_names_ok[i2]
                info = bildname2info.get(bname, {})
                top.append((float(sims2[i2]), bname, info.get("name", ""), info.get("link", ""),
                            info.get("verzeichnis", ""), info.get("vermutung", ""), info.get("ungeeignet", "")))

            basisname = os.path.splitext(os.path.basename(eingabe_pfad))[0]
            ausgabe_pfad = os.path.join(AUSGABE_DIR, basisname)
            os.makedirs(ausgabe_pfad, exist_ok=True)
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
.warn-flag .tooltip { visibility: hidden; opacity: 0; transition: opacity 0.2s;
  position: absolute; bottom: 125%; left: 0; max-width: 320px; background: #fffbea; color: #222;
  border: 1px solid #f2d024; border-radius: 6px; padding: 8px 10px; box-shadow: 0 6px 18px rgba(0,0,0,.15); z-index: 10;}
.warn-flag:hover .tooltip { visibility: visible; opacity: 1; }
.warn-flag .tooltip::after { content: ""; position: absolute; top: 100%; left: 12px; border-width: 6px; border-style: solid;
  border-color: #f2d024 transparent transparent transparent; }
</style></head><body>
<h2>Eingabebild</h2>""")
                f.write(f'<img src="{os.path.basename(eingabe_pfad)}" style="max-width:100%;height:auto;margin-bottom:20px">')
                f.write(f"<h2>Ähnlichste Bilder zu {basisname}</h2><div class='grid'>")
                for score, bildname, name, link, verzeichnis, vermutung, ungeeignet_val in top:
                    warn_html_parts = []
                    if str(vermutung).strip().lower() == "vermutung":
                        tooltip_text = (
                            f'Hintergrund: Dieses Objekt heißt "{bildname}" auf der Objektdatenbank. '
                            'Der Pfad und/oder der Name des Zusi-Objekts im offiziellen Bestand ist nicht exakt der gleiche. '
                            'Mit einer Ähnlichkeitssuche wurde dieses und weitere Objekte automatisch zugeordnet. '
                            'Das kann bei manchen Objekten falsch sein.'
                        )
                        warn_html_parts.append(
                            "<span class='warn-flag'>??"
                            f"<span class='tooltip'>{tooltip_text}</span>"
                            "</span> "
                            "<span style='color:#b00020;font-weight:600;'>Hinweis: Die Daten zu diesem Objekt sind womöglich nicht korrekt (Dateiname und Verzeichnis können abweichen).</span>"
                        )
                    if str(ungeeignet_val).strip().lower() == "ungeeignet":
                        warn_html_parts.append("<div style='color:#b00020;font-weight:600;margin-top:4px'>?? Hinweis: Dieses Objekt ist vermutlich ungeeignet für den Geländeformer.</div>")
                    warn_html = "".join(warn_html_parts)

                    src_img = finde_bild_pfad(bildname)
                    dst_img = os.path.join(ausgabe_pfad, bildname)
                    try:
                        shutil.copy2(src_img, dst_img)
                    except Exception:
                        continue

                    link_html = f"<a href='{link}' target='_blank'>Link</a><br>" if link and link != "-" else ""
                    verzeichnis_pfad = resolve_verzeichnis_path(verzeichnis, zusi_pfad)
                    if verzeichnis_pfad:
                        verzeichnis_url = 'file:///' + verzeichnis_pfad.replace('\\', '/').replace(' ', '%20')
                        # Vollstaendiger Windows-Pfad fuer Zwischenablage
                        path_for_clip = verzeichnis_pfad.replace('/', '\\')
                        # HTML-Attribut-Escaping (& < > " Zeichen)
                        import html as _html
                        path_attr = _html.escape(path_for_clip, quote=True)
                        link_objekt = (
                            f"<a href='{verzeichnis_url}' target='_blank'>Zum Objektordner</a><br>"
                            f"<button type='button' onclick='navigator.clipboard.writeText(this.dataset.pfad)' data-pfad='{path_attr}' style='cursor:pointer;border:none;background:none;color:#0066cc;padding:0;font-size:inherit;'>&#128203; Pfad kopieren</button>"
                            f"{warn_html}<br>"
                        )
                        anzeige_verzeichnis = verzeichnis_pfad.replace('/', '\\')
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
            durations.append(time.time() - current_image_start)
            ergebnisse_counter += 1

        fortschritt.set(100)
        eta_text_var.set("Fertig")
        progress_text_var.set("Analyse abgeschlossen")

    # Layout
    frm = tk.Frame(root)
    frm.pack(padx=12, pady=12)
    tk.Button(frm, text="Bilder auswählen", command=bilder_auswaehlen).grid(row=0, column=0, sticky="w")
    tk.Label(frm, textvariable=progress_text_var).grid(row=0, column=1, sticky="w", padx=8)

    tk.Label(root, text="Zusi-Stammdatenverzeichnis (Ordner _ZusiData wählen):").pack(anchor="w", padx=12, pady=(8, 0))
    tk.Entry(root, textvariable=zusi_pfad_var, width=60).pack(anchor="w", padx=12)
    tk.Button(root, text="Pfad auswählen", command=lambda: zusi_pfad_var.set(tk.filedialog.askdirectory())).pack(anchor="w", padx=12, pady=(4, 8))

    tk.Button(
        root,
        text="Embeddings (Standard) erstellen/aktualisieren",
        command=lambda: threading.Thread(target=build_embeddings, daemon=True).start()
    ).pack(anchor="w", padx=12, pady=(4, 4))

    if rembg_verfuegbar:
        tk.Checkbutton(root, text="Hintergrund mit KI entfernen (rembg)", variable=hintergrund_entfernen_var).pack(anchor="w", padx=12)
    else:
        tk.Label(root, text="Hintergrund entfernen nicht verfügbar – pip install rembg", fg="gray").pack(anchor="w", padx=12)
    tk.Checkbutton(root, text="Ergebnis sofort öffnen", variable=ergebnis_oeffnen_var).pack(anchor="w", padx=12)
    tk.Checkbutton(root, text="Verzeichnis sofort öffnen", variable=ordner_oeffnen_var).pack(anchor="w", padx=12)

    tk.Button(
        root,
        text="Analyse starten",
        command=lambda: threading.Thread(target=analyse_ausfuehren, daemon=True).start()
    ).pack(pady=8, padx=12)

    ttk.Progressbar(root, maximum=100, variable=fortschritt, length=320, mode="determinate").pack(anchor="w", padx=12)
    tk.Label(root, textvariable=eta_text_var).pack(anchor="w", padx=12, pady=(4, 0))

    root.mainloop()


if __name__ == "__main__":
    start_gui()
