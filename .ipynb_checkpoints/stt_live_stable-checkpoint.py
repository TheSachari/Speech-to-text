# stt_live_final_only.py
import argparse, queue, sys, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--model",  type=str, default="tiny")
    p.add_argument("--sr",     type=int, default=16000)
    p.add_argument("--block",  type=float, default=0.25)
    p.add_argument("--window", type=float, default=2.0)
    p.add_argument("--gpu",    action="store_true")
    p.add_argument("--stable_n", type=int, default=3)  # tours identiques avant de figer le dernier seg
    args = p.parse_args()

    device = "cuda" if args.gpu else "cpu"
    compute = "float16" if args.gpu else "int8"

    model = WhisperModel(args.model, device=device, compute_type=compute)

    q = queue.Queue()
    ring = np.zeros(int(args.sr*args.window), dtype=np.float32)

    def cb(indata, frames, time_info, status):
        if status: print(status, file=sys.stderr)
        q.put(indata[:,0].astype(np.float32).copy())

    print("🎤 Parlez… (Ctrl+C pour quitter)")
    last_committed_idx = -1          # index du dernier segment imprimé
    last_tail_text = ""              # texte du segment de queue
    tail_same_count = 0

    with sd.InputStream(channels=1, samplerate=args.sr,
                        blocksize=int(args.sr*args.block), dtype="float32",
                        device=args.device, callback=cb):
        committed_text = []
        while True:
            blk = q.get()
            n = len(blk)
            ring = np.concatenate([ring[n:], blk])

            # timestamps nécessaires pour découper en segments
            segments, _ = model.transcribe(
                ring, language="fr", beam_size=1,
                vad_filter=True, without_timestamps=False,
                condition_on_previous_text=True,
                vad_parameters={"min_silence_duration_ms": 250}
            )
            segs = list(segments)

            if not segs:
                continue

            # on considère le dernier comme "en cours"
            stable = segs[:-1]
            tail = segs[-1].text

            # imprime seulement les segments stables non encore imprimés
            if len(stable) - 1 > last_committed_idx:
                for s in stable[last_committed_idx+1:]:
                    txt = s.text.strip()
                    if txt:
                        print(txt, flush=True)
                        committed_text.append(txt)
                last_committed_idx = len(stable) - 1
                # reset suivi du tail
                last_tail_text = tail
                tail_same_count = 1
                continue

            # sinon, surveille le dernier pour le figer si stable
            if tail == last_tail_text:
                tail_same_count += 1
            else:
                last_tail_text = tail
                tail_same_count = 1

            if tail and (tail.endswith((".", "!", "?", "…")) or tail_same_count >= args.stable_n):
                print(tail.strip(), flush=True)
                committed_text.append(tail.strip())
                last_committed_idx += 1
                last_tail_text = ""
                tail_same_count = 0
