"""Generate the built-in sound cues for agent-sound-alerts.

Light unobtrusive "drums": short muted taps — a membrane tone with a fast
downward pitch drop plus a soft noise hit of the strike. No ringing, no sharp
fronts, peak volume around -12 dBFS.

Run: python scripts/make_sounds.py  (overwrites sounds/*.wav)
"""

import wave
from pathlib import Path

import numpy as np

SR = 44100
RNG = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent.parent / "sounds"


def hit(freq: float, tau: float, dur: float, tone_amp: float, click_amp: float) -> np.ndarray:
    """One soft drum hit.

    The membrane tone starts at ~1.5x the fundamental and drops quickly to it
    (the "tuk" effect), decaying exponentially with the time constant tau.
    The stick "click" noise is smoothed by averaging — muffled, not sharp.
    """
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = freq * (1.0 + 0.5 * np.exp(-t / 0.012))
    phase = 2 * np.pi * np.cumsum(f) / SR
    tone = np.sin(phase) + 0.3 * np.sin(2 * phase)
    env = np.exp(-t / tau)
    attack = int(0.003 * SR)
    env[:attack] *= np.linspace(0.0, 1.0, attack)
    noise = RNG.standard_normal(n)
    # Double smoothing: removes the HF "sand", keeps the muffled hit.
    noise = np.convolve(np.convolve(noise, np.ones(6) / 6, "same"), np.ones(6) / 6, "same")
    click = noise * np.exp(-t / 0.005)
    return tone * env * tone_amp + click * click_amp


def render(pattern, note_dur: float) -> np.ndarray:
    """pattern: [(frequency Hz, start ms, amp, tau, click_amp)]."""
    t_end = 0.010 + max(start + note_dur * 1000 for _, start, *_ in pattern)
    n = int(SR * t_end / 1000)
    buf = np.zeros(n)
    for freq, start_ms, amp, tau, click in pattern:
        i0 = int(SR * start_ms / 1000)
        seg = hit(freq, tau, note_dur, amp, click)
        buf[i0 : i0 + len(seg)] += seg
    return buf


def write_wav(path: Path, buf: np.ndarray, peak: float) -> None:
    buf = buf / np.max(np.abs(buf)) * peak
    pcm = (buf * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cues = {
        # "Done": three light taps bottom-up — dong-ding-ding.
        "task_done.wav": ([(190, 10, 1.0, 0.09, 0.10), (260, 150, 0.9, 0.07, 0.10), (330, 290, 0.7, 0.07, 0.10)], 0.35, 0.25),
        # "Attention": two low muffled hits, calm, no alarm.
        "cache_overflow.wav": ([(150, 10, 1.0, 0.10, 0.08), (115, 180, 0.9, 0.10, 0.08)], 0.40, 0.24),
        # "Question": two short light ticks a bit higher, very unobtrusive.
        "question.wav": ([(340, 10, 1.0, 0.045, 0.10), (430, 135, 0.85, 0.045, 0.10)], 0.25, 0.20),
    }
    for name, (pattern, dur, peak) in cues.items():
        buf = render(pattern, dur)
        write_wav(OUT / name, buf, peak)
        print(f"{name}: {len(buf) / SR:.2f} s, peak {20 * np.log10(peak):.1f} dBFS")


if __name__ == "__main__":
    main()
