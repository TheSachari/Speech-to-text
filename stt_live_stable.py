# stt_live_preview_record_final.py
import argparse, time, queue, sys, os, datetime
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --- downsample vers 16 kHz pour Whisper (faster-whisper attend 16k) ---
def to_16k(x, sr):
    if sr == 16000:
        return x
    try:
        from scipy.signal import resample_poly
        # facteur rationnel approx si sr multiple de 100: ex 48000 -> 16k (1/3)
        g = int(round(sr / 16000))
        if abs(sr - 16000*g) < 1:  # multiple quasi entier
            return resample_poly(x, 1, g).astype(np.float32)
        # sinon ratio générique
        L = int(round(len(x) * 16000 / sr))
        return resample_poly(x, L, len(x)).astype(np.float32)
    except Exception:
        # fallback simple: 48k -> 16k par moyenne /3 ; sinon interp linéaire
        if sr == 48000:
            n = len(x) - (len(x) % 3)
            return x[:n].reshape(-1, 3).mean(axis=1).astype(np.float32)
        t = np.linspace(0, 1, len(x), endpoint=False)
        t2 = np.linspace(0, 1, int(len(x) * 16000 / sr), endpoint=False)
        return np.interp(t2, t, x).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None, help="index micro (ex: 18=pipewire, 2=USB)")
    p.add_argument("--sr", type=int, default=48000, help="sample rate de capture")
    p.add_argument("--block", type=float, default=0.25, help="taille bloc en secondes")
    p.add_argument("--window", type=float, default=3., help="fenêtre preview (s)")
    p.add_argument("--model", type=str, default="base")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--out", type=str, default=None, help="fichier .txt de sortie")
    p.add_argument("--min_sil_ms", type=int, default=300)
    p.add_argument("--commit_timeout", type=float, default=1.0)  # juste pour le confort du preview
    p.add_argument("--min_commit", type=int, default=10)
    args = p.parse_args()

    out_path = args.out or f"transcription_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    device_fw = "cuda" if args.gpu else "cpu"
    compute = "float16" if args.gpu else "int8"
    model = WhisperModel(args.model, device=device_fw, compute_type=compute)

    q = queue.Queue()
    ring_16k = np.zeros(int(16000 * args.window), dtype=np.float32)
    audio_all_16k = []  # on stocke TOUT l'audio (version 16k) pour la passe finale

    def cb(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0].astype(np.float32)
        q.put(mono.copy())

    print(f"Sortie finale -> {out_path}")
    print("🎤 Parlez… (Ctrl+C pour sauvegarder et quitter)")
    last_preview = ""
    last_change_t = time.time()

    try:
        with sd.InputStream(channels=1, samplerate=args.sr,
                            blocksize=int(args.sr * args.block),
                            dtype="float32", device=args.device, callback=cb):
            while True:
                blk = q.get()                 # audio natif (args.sr)
                blk16 = to_16k(blk, args.sr)  # converti pour Whisper
                audio_all_16k.append(blk16)   # on garde tout pour la passe finale

                # preview: fenêtre glissante 16k
                n = len(blk16)
                ring_16k = np.concatenate([ring_16k[n:], blk16])

                seg_it, _ = model.transcribe(
                    ring_16k,
                    language="fr",
                    beam_size=1,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": args.min_sil_ms, "speech_pad_ms": 180},
                    without_timestamps=False,
                    condition_on_previous_text=False,   # preview + stable
                    initial_prompt=None,                # ou une courte liste de mots-clés
                )

                segs = list(seg_it)
                if not segs:
                    continue

                # on affiche seulement le dernier segment comme preview (réécrit la ligne)
                tail = segs[-1].text.strip()
                if tail != last_preview:
                    last_preview = tail
                    last_change_t = time.time()
                if tail:
                    sys.stdout.write("\r" + tail + " " * 8)
                    sys.stdout.flush()

                # (optionnel) sauter de ligne si on détecte une ponctuation/stabilité -> plus lisible
                if tail and (tail.endswith((".", "!", "?", "…")) or
                             (time.time() - last_change_t >= args.commit_timeout and len(tail) >= args.min_commit)):
                    print("\r" + tail + " " * 8)

    except KeyboardInterrupt:
        pass
    finally:
        # === Passe finale sur tout l'audio cumulé (UNE transcription propre) ===
        if audio_all_16k:
            full = np.concatenate(audio_all_16k).astype(np.float32)
            seg_it, _ = model.transcribe(
                full,
                language="fr",
                beam_size=6,               # un peu plus soigné pour le rendu final
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": args.min_sil_ms, "speech_pad_ms": 160},
                without_timestamps=False,
                condition_on_previous_text=True,
            )
            final_lines = [s.text.strip() for s in seg_it if s.text and s.text.strip()]
            text = "\n".join(final_lines).strip() + ("\n" if final_lines else "")
        else:
            text = ""

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n✅ Transcription enregistrée (passe finale) : {out_path}")

if __name__ == "__main__":
    main()

