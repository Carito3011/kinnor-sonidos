"""
Kinnor Orquesta 2 — librería grande de orquesta (se descarga aparte, no va dentro de la app).

Junta las mejores librerías libres de orquesta:
  - VS Chamber Orchestra 2 Community Edition (CC0)                       https://vis.versilstudios.net/vsco-2.html
  - Sonatina Symphonic Orchestra (Creative Commons Sampling Plus 1.0)    http://sso.mattiaswestlund.net
  - Partes de Virtual Playing Orchestra 3 (No Budget Orchestra CC BY-SA 4.0, Mattias Westlund CC BY-SA 3.0,
    Philharmonia Orchestra CC BY-SA 3.0, University of Iowa MIS "sin restricciones")  https://virtualplaying.com

Calidad completa: estéreo, 44,1 kHz, 16 bits, notas largas con bucle largo (para que conserven el vibrato).
Cada sonido tiene:
  banco 210  sonido normal (con capas de intensidad cuando la librería las trae: suave / medio / fuerte)
  banco 211  gemelo "legato" (sin ataque) que el motor usa para notas ligadas
  banco 212  articulación corta (staccato / spiccato) que el motor usa solo para las notas cortas
Los números de programa son los mismos de Kinnor Orquesta 1 (0–33), así los proyectos viejos suenan con la nueva.

Uso: python3 tools/build_orquesta2.py VSCO SSO VPO_LIBS ORQUESTA1.sf2 salida.sf2
"""
import glob
import math
import os
import re
import sys
import warnings
from collections import Counter

import numpy as np
import soundfile
from scipy.signal import resample_poly

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from sf2 import SF2, GEN as G, rng, u16  # noqa: E402

VSCO, SSO, VPO, OLD, DST = sys.argv[1:6]
SSO = os.path.join(SSO, 'Sonatina Symphonic Orchestra', 'Samples')
RATE = 44100
BANK, LEG_BANK, SHORT_BANK = 210, 211, 212

NAMES = {'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4, 'F': 5, 'F#': 6, 'GB': 6, 'G': 7, 'G#': 8,
         'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11}
TOK = re.compile(r'^([A-Ga-g])([#b]?)(-?\d)$')
OCT_FIRST = re.compile(r'(?:^|[_-])(\d)_([A-Ga-g][b#]?)(?=[_.-]|$)')


def tc(seconds):
    return u16(round(1200 * math.log2(max(seconds, 0.001))))


def note_of(path):
    base = os.path.splitext(os.path.basename(path))[0]
    for t in re.split(r'[-_ .]', base):
        m = TOK.match(t)
        if m:
            return (int(m.group(3)) + 1) * 12 + NAMES[(m.group(1) + m.group(2)).upper()]
    m = OCT_FIRST.search(base)
    if m:
        return (int(m.group(1)) + 1) * 12 + NAMES[m.group(2).upper()]
    return None


def load(path):
    x, rate = soundfile.read(path, always_2d=True, dtype='float64')
    x = x[:, :2]
    if rate != RATE:
        g = math.gcd(rate, RATE)
        x = resample_poly(x, RATE // g, rate // g, axis=0)
    return x


def trim_start(x):
    mono = np.abs(x).max(axis=1)
    peak = mono.max() or 1.0
    idx = int(np.argmax(mono > peak * 0.02))
    return x[max(0, idx - int(0.004 * RATE)):]


def detect_midi(m, start):
    seg = m[int(start * RATE):int((start + 0.6) * RATE)]
    if len(seg) < 4096:
        seg = m[:min(len(m), 16384)]
    seg = seg - seg.mean()
    n = len(seg)
    if n < 1024:
        return None
    f = np.fft.rfft(seg, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    lo, hi = int(RATE / 2200), int(RATE / 25)
    if hi >= n:
        return None
    k = lo + int(np.argmax(ac[lo:hi]))
    if ac[k] <= 0:
        return None
    if 1 <= k < n - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        d = (a - c) / (2 * (a - 2 * b + c)) if (a - 2 * b + c) != 0 else 0
        k = k + d
    if k <= 1:
        return None
    return 69 + 12 * math.log2(RATE / k / 440.0)


def make_loop(x):
    """Bucle largo con fundido (conserva el vibrato natural)."""
    dur = len(x) / RATE
    le = min(dur - 0.12, 6.0)
    ls = min(max(0.9, le * 0.3), le - 0.6)
    if ls < 0.15:   # muestra muy corta: bucle con todo lo que hay
        ls, le = dur * 0.35, dur * 0.95
    xf = min(0.35, (le - ls) / 3, ls * 0.9)
    a, b, w = int(ls * RATE), int(le * RATE), max(32, int(xf * RATE))
    y = x[:b].copy()
    t = np.linspace(0, 1, w)[:, None]
    y[b - w:b] = y[b - w:b] * np.cos(t * np.pi / 2) + x[a - w:a] * np.sin(t * np.pi / 2)
    return y, a, b


def pcm(ch):
    return (np.clip(ch, -1, 1) * 32767).astype('<i2').tobytes()


sf = SF2.empty('Kinnor Orquesta 2',
               'VSCO 2 CE (CC0), Sonatina Symphonic Orchestra (CC Sampling Plus 1.0), Virtual Playing Orchestra 3 '
               '(No Budget Orchestra, Mattias Westlund, Philharmonia Orchestra CC BY-SA; Univ. of Iowa MIS). Armado por Kinnor.')


class L:
    """Una fuente de muestras dentro de un instrumento."""

    def __init__(self, pattern, vel=(1, 127), db=0.0, exclude=None, spread=True, octave=None, every=1):
        self.pattern, self.vel, self.db, self.exclude, self.spread, self.octave, self.every = \
            pattern, vel, db, exclude, spread, octave, every


TOTAL = [0]


def build(name, layers, sustain=True, attack=0.08, release=0.5, target=-18.0, max_len=3.0, stereo=True):
    """Instrumento SF2 con varias fuentes (capas de intensidad o capas apiladas). Devuelve (inst, gemelo legato)."""
    zones = []
    first = True
    for lay in layers:
        files = sorted(f for f in glob.glob(lay.pattern) if f.lower().endswith(('.wav', '.flac')))
        if lay.exclude:
            files = [f for f in files if not re.search(lay.exclude, os.path.basename(f))]
        if not files:
            raise SystemExit('Sin muestras: %s  %s' % (name, lay.pattern))
        loaded = {}
        offsets = []
        for f in files:
            x = trim_start(load(f))
            if len(x) < int(0.05 * RATE):
                continue
            m = x.mean(axis=1)
            det = detect_midi(m, 0.35 if sustain else 0.03)
            midi = note_of(f)
            if midi is None:
                if det is None:
                    continue
                midi = int(round(det))
            elif det is not None:
                offsets.append(round((det - midi) / 12) * 12)
            if midi in loaded:  # nota repetida (round robin): se queda la primera
                continue
            loaded[midi] = (x, det)
        fix = lay.octave if lay.octave is not None else (Counter(offsets).most_common(1)[0][0] if offsets else 0)
        items = sorted((k + fix, v) for k, v in loaded.items())[::lay.every]
        notes = [k for k, _ in items]
        for i, (root, (x, det)) in enumerate(items):
            if lay.spread:
                lo = 0 if i == 0 else (notes[i - 1] + root) // 2 + 1
                hi = 127 if i == len(items) - 1 else (root + notes[i + 1]) // 2
            else:  # capa apilada: solo cerca de sus notas
                lo = max(0, root - 2) if i == 0 else (notes[i - 1] + root) // 2 + 1
                hi = min(127, root + 2) if i == len(items) - 1 else (root + notes[i + 1]) // 2
            cents = 0
            if det is not None and abs(det - root) < 0.6:
                cents = -int(round((det - root) * 100))
            if sustain:
                y, a, b = make_loop(x)
                ref = y[int(0.3 * RATE):a].mean(axis=1) if a > int(0.5 * RATE) else y.mean(axis=1)
            else:
                y = x[:int(max_len * RATE)].copy()
                fade = min(len(y) // 4, int(0.1 * RATE))
                if fade > 0:
                    y[-fade:] *= np.linspace(1, 0, fade)[:, None]
                a = b = None
                ref = y[:int(0.35 * RATE)].mean(axis=1)
            rms = math.sqrt(float(np.mean(ref ** 2))) or 1e-6
            y = y * (10 ** ((target + lay.db) / 20) / rms)
            peak = float(np.max(np.abs(y)))
            if peak > 0.98:
                y *= 0.98 / peak
            chans = [y[:, 0], y[:, 1]] if (stereo and y.shape[1] == 2) else [y.mean(axis=1)]
            for ci, ch in enumerate(chans):
                sid = sf.add_sample('%s %d%s' % (name[:12], root, 'LR'[ci] if len(chans) == 2 else ''), pcm(ch), RATE,
                                    root, loop=(a, b) if sustain else None, correction=max(-99, min(99, cents)))
                TOTAL[0] += len(ch) * 2
                z = [(G['keyRange'], rng(lo, hi)), (G['velRange'], rng(*lay.vel)), (G['sampleModes'], 1 if sustain else 0),
                     (G['attackVolEnv'], tc(attack)), (G['releaseVolEnv'], tc(release)), (G['reverb'], 120)]
                if len(chans) == 2:
                    z.append((G['pan'], u16(-500 if ci == 0 else 500)))
                z.append((G['sampleID'], sid))
                zones.append(z)
        print('  %-20s %-60s %2d notas  oct %+d' % (name, lay.pattern[-60:], len(items), fix))
        first = False
    iid = sf.add_instrument(name[:20], zones)
    leg = None
    if sustain:
        lz = []
        for z in zones:
            sid = dict(z)[G['sampleID']]
            h = sf.rec['shdr'][sid]
            off = max(0, min(int(0.2 * RATE), h[3] - h[1] - int(0.05 * RATE)))
            nz = [g for g in z if g[0] not in (G['attackVolEnv'], G['sampleID'])]
            nz += [(G['attackVolEnv'], tc(0.1)), (G['startAddrsOffset'], u16(off)), (G['sampleID'], sid)]
            lz.append(nz)
        leg = sf.add_instrument((name[:17] + ' L')[:20], lz)
    print('%-22s listo — %d MB acumulados' % (name, TOTAL[0] // 1_000_000))
    return iid, leg


J = os.path.join
S, V, P = (lambda *a: J(SSO, *a)), (lambda *a: J(VSCO, *a)), (lambda *a: J(VPO, *a))
I, LEG, SH = {}, {}, {}


def inst(key, name, layers, short=None, short_target=-15.0, **kw):
    I[key], LEG[key] = build(name, layers, **kw)
    if short:
        SH[key], _ = build(name[:14] + ' corto', short, sustain=False, attack=0.002, release=0.25, max_len=1.2,
                           target=short_target)


def shot(key, name, layers, **kw):
    kw.setdefault('sustain', False)
    I[key], LEG[key] = build(name, layers, **kw)


STR = dict(attack=0.09, release=0.5)
# ---------- Cuerdas ----------
inst('vln', 'Violines', [L(S('1st Violins', '*-sus-*')), L(V('Strings', 'Violin Section', 'susVib', '*'), db=-2, spread=False)],
     short=[L(S('1st Violins', '*-stc-rr1-*')), L(V('Strings', 'Violin Section', 'Spic', '*'), db=-2, spread=False)], **STR)
inst('vln2', 'Violines 2', [L(S('2nd Violins', '*-sus-*')), L(P('Other', 'Samples', '2nd-Violin-Section', '*'), db=-3, spread=False)],
     short=[L(S('2nd Violins', '*-stc-rr1-*'))], **STR)
inst('vla', 'Violas', [L(S('Violas', 'violas-sus*')), L(V('Strings', 'Viola Section', 'susvib', '*'), db=-2, spread=False),
                       L(P('Mattias-Westlund', 'ViolaSect', 'Samples', '*'), db=-4, spread=False)],
     short=[L(S('Violas', '*-stc-rr1-*'))], **STR)
inst('vc', 'Cellos', [L(S('Celli', 'celli-sus*')), L(V('Strings', 'Cello Section', 'susvib', '*'), db=-2, spread=False),
                      L(P('NoBudgetOrch', 'CelloSect', 'Sustain', '*'), db=-4, spread=False)],
     short=[L(S('Celli', 'celli-stc-rr1*')), L(P('VSCO2-CE', 'Strings', 'Cello Section', 'spic', '*'), db=-3, spread=False)], **STR)
inst('cb', 'Contrabajos', [L(S('Basses', 'basses-sus*')), L(P('Other', 'Samples', 'Bass-Section', '*'), db=-3, spread=False)],
     short=[L(S('Basses', 'basses-stc-rr1*'))], **STR)
inst('trem', 'Trémolo', [L(S('1st Violins', '*-trm-*')), L(V('Strings', 'Violin Section', 'Trem', '*_v2*'), db=-2, spread=False)],
     attack=0.04, release=0.4)
shot('pizz', 'Pizzicato', [L(S('1st Violins', '*-piz-rr1-*')), L(V('Strings', 'Violin Section', 'Pizz', '*'), db=-3, spread=False)],
     attack=0.001, release=0.3, max_len=1.5, target=-14)
shot('spic', 'Spiccato', [L(S('1st Violins', '*-stc-rr1-*')), L(V('Strings', 'Violin Section', 'Spic', '*'), db=-2, spread=False)],
     attack=0.002, release=0.25, max_len=1.2, target=-15)
inst('vlnSolo', 'Violín Solo', [L(V('Strings', 'Solo Violin', 'Arco Vib', '*')), L(S('Violin 2', 'violin_arco_vib*'), db=-3, spread=False)],
     short=[L(S('Violin 2', 'violin_spiccato*'))], attack=0.07, release=0.4)
shot('vlnSoloPizz', 'Violín Solo Pizz', [L(S('Violin 2', 'violin_pizz_vib*')), L(V('Strings', 'Solo Violin', 'Pizz', '*'), db=-3, spread=False)],
     attack=0.001, release=0.3, max_len=1.5, target=-14)
inst('vlaSolo', 'Viola Solo', [L(S('Viola', 'viola-sus*')), L(P('NoBudgetOrch', 'SoloViola', 'Vibrato', '*'), db=-3, spread=False)],
     attack=0.07, release=0.4)
inst('vcSolo', 'Cello Solo', [L(S('Cello', 'cello-[a-g]*'), exclude='piz'), L(P('NoBudgetOrch', 'SoloCello', 'Vibrato', '*'), db=-3, spread=False)],
     attack=0.07, release=0.45)
shot('vcPizz', 'Cellos Pizzicato', [L(S('Celli', 'celli-piz-rr1*')), L(V('Strings', 'Cello Section', 'pizzT', '*'), db=-3, spread=False)],
     attack=0.001, release=0.3, max_len=1.8, target=-14)
shot('cbPizz', 'Contrabajo Pizz', [L(S('Basses', 'basses-piz-rr1*')), L(V('Strings', 'Solo Contrabass', 'Pizz', '*'), db=-3, spread=False)],
     attack=0.001, release=0.3, max_len=2.0, target=-14)
shot('harp', 'Arpa', [L(S('Harp', '*')), L(V('Strings', 'Harp', '*'), db=-3, spread=False)], attack=0.001, release=0.6, max_len=4.0, target=-15)

# ---------- Metales (con capas de intensidad) ----------
BR = dict(attack=0.06, release=0.4)
inst('hn', 'Cornos', [L(S('Horns', 'horns-sus-mp*'), vel=(1, 80)), L(S('Horns', 'horns-sus-ff*'), vel=(81, 127)),
                      L(V('Brass', 'F Horn', 'sus', '*'), db=-3, spread=False)],
     short=[L(S('Horns', 'horns-stc-rr1*'))], **BR)
inst('tpt', 'Trompeta', [L(V('Brass', 'Trumpet', 'sus', '*_v1_*'), vel=(1, 70)), L(V('Brass', 'Trumpet', 'sus', '*_v3_*'), vel=(71, 127)),
                         L(S('Trumpet', 'trumpet*'), db=-3, spread=False)],
     short=[L(S('Trumpets', 'trumpets-stc-rr1*'))], **BR)
inst('tbn', 'Trombón', [L(V('Brass', 'Tenor Trombone', 'sus', '*_v1_*'), vel=(1, 70)), L(V('Brass', 'Tenor Trombone', 'sus', '*_v2_*'), vel=(71, 127)),
                        L(S('Trombones', 'trombones-sus*'), db=-3, spread=False)],
     short=[L(S('Trombones', 'trombones-stc-rr1*'))], **BR)
inst('tba', 'Tuba', [L(V('Brass', 'Tuba', 'sus', '*_v1_*'), vel=(1, 55)), L(V('Brass', 'Tuba', 'sus', '*_v2_*'), vel=(56, 95)),
                     L(V('Brass', 'Tuba', 'sus', '*_v3_*'), vel=(96, 127)), L(S('Tuba', 'tuba-sus*'), db=-3, spread=False)],
     short=[L(S('Tuba', 'tuba-stc-rr1*'))], **BR)
inst('tpts', 'Trompetas', [L(S('Trumpets', 'trumpets-sus*'))], short=[L(S('Trumpets', 'trumpets-stc-rr1*'))], **BR)
inst('btbn', 'Trombón Bajo', [L(S('Bass Trombone', '*'))], **BR)

# ---------- Maderas ----------
WW = dict(attack=0.05, release=0.35)
inst('fl', 'Flauta', [L(S('Flute 2', 'fl2_sus_vb_mf*'), vel=(1, 80)), L(S('Flute 2', 'fl2_sus_vb_ff*'), vel=(81, 127)),
                      L(V('Woodwinds', 'Flute', 'susvib', '*'), db=-3, spread=False)],
     short=[L(S('Flute 2', 'fl2_stc_mf_rr1*'), vel=(1, 80)), L(S('Flute 2', 'fl2_stc_ff_rr1*'), vel=(81, 127))], **WW)
inst('ob', 'Oboe', [L(V('Woodwinds', 'Oboe', 'Vib', '*_v1_*'), vel=(1, 70)), L(V('Woodwinds', 'Oboe', 'Vib', '*_v3_*'), vel=(71, 127)),
                    L(S('Oboe', '*'), db=-3, spread=False)], **WW)
inst('cl', 'Clarinete', [L(V('Woodwinds', 'Clarinet', 'susLong', '*_v1_*'), vel=(1, 55)), L(V('Woodwinds', 'Clarinet', 'susLong', '*_v2_*'), vel=(56, 95)),
                         L(V('Woodwinds', 'Clarinet', 'susLong', '*_v3_*'), vel=(96, 127)), L(S('Clarinet', '*'), db=-3, spread=False)], **WW)
inst('fg', 'Fagot', [L(V('Woodwinds', 'Bassoon', 'sus', '*_v1_*'), vel=(1, 70)), L(V('Woodwinds', 'Bassoon', 'sus', '*_v2_*'), vel=(71, 127)),
                     L(S('Bassoon', '*'), db=-3, spread=False)], **WW)
inst('picc', 'Flautín', [L(S('Piccolo', '*')), L(V('Woodwinds', 'Piccolo', 'Sus', '*'), db=-3, spread=False)], **WW)
inst('eh', 'Corno Inglés', [L(S('Cor Anglais', '*')), L(P('Philharmonia', 'Samples', 'English-Horn', '*'), db=-3, spread=False)], **WW)
inst('bcl', 'Clarinete Bajo', [L(S('Bass Clarinet', '*'))], **WW)
inst('cfg', 'Contrafagot', [L(S('Contrabassoon', '*'))], **WW)
inst('afl', 'Flauta Alto', [L(S('Alto Flute', '*'))], **WW)

# ---------- Teclados y percusión afinada ----------
shot('glk', 'Glockenspiel', [L(S('Percussion', 'glockenspiel*'))], attack=0.001, release=0.8, max_len=3.0, target=-16)
shot('timp', 'Timbales', [L(S('Percussion', 'timpani-p-rh*'), vel=(1, 75)), L(S('Percussion', 'timpani-f-rh*'), vel=(76, 127))],
     attack=0.001, release=0.8, max_len=4.0, target=-13)
inst('timpRoll', 'Timbales Redoble', [L(S('Percussion', 'timpani-roll-[a-g]*'))], attack=0.08, release=0.6)
shot('mar', 'Marimba', [L(S('Marimba', 'marimba-yarn-pp*'), vel=(1, 50)), L(S('Marimba', 'marimba-yarn-mf*'), vel=(51, 95)),
                        L(S('Marimba', 'marimba-yarn-ff*'), vel=(96, 127))], attack=0.001, release=0.6, max_len=2.5, target=-15)
shot('xyl', 'Xilófono', [L(S('Percussion', 'xylophone*')), L(P('NoBudgetOrch', 'Percussion', 'Xylophone', '*'), db=-3, spread=False)],
     attack=0.001, release=0.4, max_len=1.5, target=-15)
shot('cel', 'Celesta', [L(S('Celeste', '*-soft*'), vel=(1, 75)), L(S('Celeste', '*-hard*'), vel=(76, 127))],
     attack=0.001, release=0.7, max_len=3.0, target=-15)
shot('vib', 'Vibráfono', [L(S('Vibraphone', '*'))], attack=0.001, release=0.9, max_len=4.0, target=-16)
shot('chimes', 'Campanas', [L(S('Percussion', 'chimes-*')), L(P('NoBudgetOrch', 'Percussion', 'TubularBells', '*'), db=-3, spread=False)],
     attack=0.001, release=1.2, max_len=5.0, target=-15)
inst('coro', 'Coro', [L(S('Chorus', 'chorus-female*')), L(S('Chorus', 'chorus-male*'))], attack=0.15, release=0.7)

# ---------- Los de Kinnor Orquesta 1 que se conservan (kits de percusión y órgano) ----------
old = SF2(OLD)
_inst_map, _smp_map = {}, {}


def copy_sample(sid):
    if sid not in _smp_map:
        h = old.rec['shdr'][sid]
        start, end, ls, le, rate, root, corr = h[1], h[2], h[3], h[4], h[5], h[6], h[7]
        data = bytes(old.smpl[start * 2:end * 2])
        loop = (ls - start, le - start) if le > ls else None
        _smp_map[sid] = sf.add_sample(h[0].split(b'\x00')[0].decode('latin1'), data, rate, root, loop=loop, correction=corr)
    return _smp_map[sid]


def copy_instrument(iid):
    if iid not in _inst_map:
        rec = old.rec['inst']
        b0 = rec[iid][1]
        b1 = rec[iid + 1][1] if iid + 1 < len(rec) else len(old.rec['ibag'])
        zones = []
        for b in range(b0, b1):
            g0 = old.rec['ibag'][b][0]
            g1 = old.rec['ibag'][b + 1][0] if b + 1 < len(old.rec['ibag']) else len(old.rec['igen'])
            z = [tuple(old.rec['igen'][g]) for g in range(g0, g1)]
            z = [(o, copy_sample(a)) if o == G['sampleID'] else (o, a) for o, a in z]
            zones.append(z)
        _inst_map[iid] = sf.add_instrument(rec[iid][0].split(b'\x00')[0].decode('latin1'), zones)
    return _inst_map[iid]


def old_zones(program):
    idx = old.find_preset(BANK, program)
    return [[(o, copy_instrument(a)) if o == G['instrument'] else (o, a) for o, a in z] for z in old.preset_zones(idx)]


# ---------- Sonidos (mismo número de programa que Kinnor Orquesta 1) ----------
def z(key, lo=0, hi=127, atten=0, pan=0, tune=0, inst_map=None, attack_add=None, release_add=None):
    zz = [(G['keyRange'], rng(lo, hi))]
    if atten:
        zz.append((G['initialAttenuation'], atten))
    if pan:
        zz.append((G['pan'], u16(pan)))
    if tune:
        zz.append((G['coarseTune'], u16(tune)))
    if attack_add:
        zz.append((G['attackVolEnv'], tc(attack_add)))
    if release_add:
        zz.append((G['releaseVolEnv'], tc(release_add)))
    zz.append((G['instrument'], (inst_map or I)[key]))
    return zz


SLOW, LONG = 0.5, 1.2
PRESETS = [
    ('Violines', [z('vln')]),
    ('Violas', [z('vla')]),
    ('Cellos', [z('vc')]),
    ('Contrabajos', [z('cb'), z('cb', atten=60, tune=-12)]),
    ('Cuerdas Ensemble', [z('vln', 55, 127, pan=-180), z('vla', 48, 84, atten=20, pan=-60), z('vc', 36, 72, atten=20, pan=90),
                          z('cb', 0, 52, atten=30, pan=160)]),
    ('Violines Pizzicato', [z('pizz')]),
    ('Violines Trémolo', [z('trem')]),
    ('Violines Spiccato', [z('spic')]),
    ('Arpa', [z('harp')]),
    ('Cornos', [z('hn')]),
    ('Trompeta', [z('tpt')]),
    ('Trombón', [z('tbn')]),
    ('Tuba', [z('tba')]),
    ('Metales Sección', [z('tpts', 60, 127, pan=-120), z('hn', 48, 84, atten=20, pan=60), z('tbn', 36, 66, atten=20, pan=120),
                         z('tba', 0, 50, atten=30)]),
    ('Flauta', [z('fl')]),
    ('Oboe', [z('ob')]),
    ('Clarinete', [z('cl')]),
    ('Fagot', [z('fg')]),
    ('Maderas Sección', [z('fl', 72, 127, pan=-120), z('ob', 60, 90, atten=20, pan=-40), z('cl', 48, 80, atten=20, pan=60),
                         z('fg', 0, 60, atten=20, pan=120)]),
    ('Glockenspiel', [z('glk')]),
    ('Percusión Orquesta', None),   # del kit de Kinnor Orquesta 1
    ('Cinematic Vast', [z('vln', 55, 127, attack_add=SLOW, release_add=LONG, pan=-180),
                        z('vla', 48, 84, atten=20, attack_add=SLOW, release_add=LONG, pan=-60),
                        z('vc', 36, 72, atten=20, attack_add=SLOW, release_add=LONG, pan=90),
                        z('cb', 0, 52, atten=20, attack_add=SLOW, release_add=LONG, pan=160),
                        z('hn', 41, 77, atten=60, attack_add=SLOW, release_add=LONG),
                        z('cb', 0, 64, atten=70, tune=-12, attack_add=SLOW, release_add=LONG)]),
    ('Violín Solo', [z('vlnSolo')]),
    ('Violín Solo Pizz', [z('vlnSoloPizz')]),
    ('Cellos Pizzicato', [z('vcPizz')]),
    ('Contrabajo Pizz', [z('cbPizz')]),
    ('Flautín', [z('picc')]),
    ('Timbales', [z('timp')]),
    ('Timbales Redoble', [z('timpRoll')]),
    ('Marimba', [z('mar')]),
    ('Xilófono', [z('xyl')]),
    ('Órgano', None),
    ('Perc. Sinfónica', None),
    ('Perc. Épica', None),
    # ----- nuevos en Orquesta 2 -----
    ('Violines 2', [z('vln2')]),
    ('Viola Solo', [z('vlaSolo')]),
    ('Cello Solo', [z('vcSolo')]),
    ('Trompetas', [z('tpts')]),
    ('Trombón Bajo', [z('btbn')]),
    ('Corno Inglés', [z('eh')]),
    ('Clarinete Bajo', [z('bcl')]),
    ('Contrafagot', [z('cfg')]),
    ('Flauta Alto', [z('afl')]),
    ('Celesta', [z('cel')]),
    ('Vibráfono', [z('vib')]),
    ('Campanas', [z('chimes')]),
    ('Coro', [z('coro')]),
]
legato, short = 0, 0
for k, (name, zones) in enumerate(PRESETS):
    assert len(name.encode()) <= 20, name
    if zones is None:
        zones = old_zones(k)
    sf.add_preset(name, BANK, k, zones)
for k, (name, zones) in enumerate(PRESETS):
    if zones is None:
        continue
    keys = [next(key for key, v in I.items() if v == dict(zz)[G['instrument']]) for zz in zones]
    if all(LEG.get(key) is not None for key in keys):
        sf.add_preset(name, LEG_BANK, k, [z(key, inst_map=LEG) if False else
                                         [(o, LEG[key]) if o == G['instrument'] else (o, a) for o, a in zz]
                                         for key, zz in zip(keys, zones)])
        legato += 1
    if all(key in SH for key in keys):
        sf.add_preset(name, SHORT_BANK, k, [[(o, SH[key]) if o == G['instrument'] else (o, a) for o, a in zz]
                                            for key, zz in zip(keys, zones)])
        short += 1
print('Legato: %d sonidos · Cortos: %d sonidos' % (legato, short))
sf.save(DST)
print('Kinnor Orquesta 2: %d sonidos, %d MB' % (len(PRESETS), os.path.getsize(DST) // 1_000_000))
