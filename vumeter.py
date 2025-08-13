# file: vumeter.py
import argparse, sys, time
import numpy as np
import sounddevice as sd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=None, help="index du micro (ex: 2)")
    p.add_argument("--sr", type=int, default=16000, help="sample rate")
    p.add_argument("--block", type=float, default=0.2, help="taille bloc en secondes")
    args = p.parse_args()

    sr = args.sr
    blocksize = int(sr * args.block)

    peak_dbfs = -120.0

    def callback(indata, frames, time_info, status):
        nonlocal peak_dbfs
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0].astype(np.float32)
        rms = np.sqrt(np.mean(mono**2) + 1e-12)
        dbfs = 20 * np.log10(rms + 1e-12)  # ~[-inf, 0]
        peak_dbfs = max(peak_dbfs * 0.95, dbfs)  # petit hold

        width = 40
        # échelle grossière: -60 dB = silence, 0 dB = plein
        level = int(np.clip((dbfs + 60) / 60 * width, 0, width))
        peak = int(np.clip((peak_dbfs + 60) / 60 * width, 0, width))
        bar = "▮" * level
        bar = bar.ljust(width)
        # marqueur de pic
        if 0 <= peak < width:
            bar = bar[:peak] + "│" + bar[peak+1:]
        print(f"\r[{bar}] {dbfs:6.1f} dBFS", end="", flush=True)

    print("Liste des devices (pour mémoire) :")
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:2d}  {d['name']:<30} IN={d['max_input_channels']}  OUT={d['max_output_channels']}")
    print("\nChoisi --device avec un index IN>0 (Ctrl+C pour quitter).")
    try:
        with sd.InputStream(channels=1,
                            samplerate=sr,
                            blocksize=blocksize,
                            dtype="float32",
                            device=args.device,
                            callback=callback):
            while True:
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nBye.")
    except Exception as e:
        print(f"\nErreur: {e}\n"
              f"Astuce: essaie --sr 44100 ou --sr 48000, ou change --device.", file=sys.stderr)

if __name__ == "__main__":
    main()
