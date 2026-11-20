"""Lectura y escritura mínima de SoundFont 2 (para agregar sonidos Kinnor al SF2 de GeneralUser)."""
import struct

GEN = dict(startAddrsOffset=0, modLfoToPitch=5, vibLfoToPitch=6, modEnvToPitch=7, initialFilterFc=8, initialFilterQ=9,
           modLfoToFilterFc=10, modEnvToFilterFc=11, modLfoToVolume=13, chorus=15, reverb=16, pan=17,
           delayModLFO=21, freqModLFO=22, delayVibLFO=23, freqVibLFO=24, delayModEnv=25, attackModEnv=26, holdModEnv=27,
           decayModEnv=28, sustainModEnv=29, releaseModEnv=30, delayVolEnv=33, attackVolEnv=34, holdVolEnv=35,
           decayVolEnv=36, sustainVolEnv=37, releaseVolEnv=38, instrument=41, keyRange=43, velRange=44,
           initialAttenuation=48, coarseTune=51, fineTune=52, sampleID=53, sampleModes=54, scaleTuning=56,
           exclusiveClass=57, overridingRootKey=58)


def chunks(data, off, end):
    while off < end:
        cid = data[off:off + 4].decode('latin1')
        size = struct.unpack_from('<I', data, off + 4)[0]
        yield cid, off + 8, size
        off += 8 + size + (size & 1)


class SF2:
    @classmethod
    def empty(cls, name, copyright_text=''):
        """SoundFont vacío para llenarlo desde cero."""
        self = cls.__new__(cls)

        def sub(cid, payload):
            if len(payload) & 1:
                payload += b'\x00'
            return cid.encode() + struct.pack('<I', len(payload)) + payload
        info = sub('ifil', struct.pack('<HH', 2, 1)) + sub('isng', b'EMU8000\x00') + sub('INAM', name.encode() + b'\x00')
        if copyright_text:
            info += sub('ICOP', copyright_text.encode() + b'\x00')
        self.info = info
        self.smpl = bytearray()
        self.sm24 = None
        self.fmt = dict(phdr=('<20sHHHIII', 38), pbag=('<HH', 4), pmod=('<HHhHH', 10), pgen=('<HH', 4),
                        inst=('<20sH', 22), ibag=('<HH', 4), imod=('<HHhHH', 10), igen=('<HH', 4),
                        shdr=('<20sIIIIIBbHH', 46))
        self.rec = {k: [] for k in self.fmt}
        self.term = dict(pmod=[0, 0, 0, 0, 0], imod=[0, 0, 0, 0, 0], pgen=[0, 0], igen=[0, 0])
        return self

    def __init__(self, path):
        data = open(path, 'rb').read()
        assert data[:4] == b'RIFF' and data[8:12] == b'sfbk'
        self.info = b''
        self.smpl = b''
        self.sm24 = None
        self.pdta = {}
        for cid, off, size in chunks(data, 12, len(data)):
            assert cid == 'LIST'
            kind = data[off:off + 4]
            if kind == b'INFO':
                self.info = data[off + 4:off + size]
            elif kind == b'sdta':
                for c2, o2, s2 in chunks(data, off + 4, off + size):
                    if c2 == 'smpl':
                        self.smpl = bytearray(data[o2:o2 + s2])
                    elif c2 == 'sm24':
                        self.sm24 = data[o2:o2 + s2]
            elif kind == b'pdta':
                for c2, o2, s2 in chunks(data, off + 4, off + size):
                    self.pdta[c2] = data[o2:o2 + s2]
        fmt = dict(phdr=('<20sHHHIII', 38), pbag=('<HH', 4), pmod=('<HHhHH', 10), pgen=('<HH', 4),
                   inst=('<20sH', 22), ibag=('<HH', 4), imod=('<HHhHH', 10), igen=('<HH', 4),
                   shdr=('<20sIIIIIBbHH', 46))
        self.fmt = fmt
        self.rec = {k: [list(struct.unpack_from(f, self.pdta[k], i)) for i in range(0, len(self.pdta[k]), n)] for k, (f, n) in fmt.items()}
        # Se quitan los registros finales (EOP/EOI/EOS); se vuelven a poner al escribir
        self.term = {k: self.rec[k].pop() for k in ('phdr', 'inst', 'shdr')}
        for k in ('pbag', 'ibag'):
            self.term[k] = self.rec[k].pop()
        for k in ('pgen', 'igen', 'pmod', 'imod'):
            self.term[k] = self.rec[k].pop() if self.rec[k] else [0] * (2 if 'gen' in k else 5)

    # ---------- lectura de zonas ----------
    def preset_zones(self, index):
        """Zonas de un preset: lista de listas [(oper, amount_u16)]."""
        p = self.rec['phdr'][index]
        b0 = p[3]
        b1 = self.rec['phdr'][index + 1][3] if index + 1 < len(self.rec['phdr']) else len(self.rec['pbag'])
        zones = []
        for b in range(b0, b1):
            g0 = self.rec['pbag'][b][0]
            g1 = self.rec['pbag'][b + 1][0] if b + 1 < len(self.rec['pbag']) else len(self.rec['pgen'])
            zones.append([tuple(self.rec['pgen'][g]) for g in range(g0, g1)])
        return zones

    def find_preset(self, bank, program):
        for i, p in enumerate(self.rec['phdr']):
            if p[1] == program and p[2] == bank:
                return i
        return None

    # ---------- agregar ----------
    def add_sample(self, name, pcm16, sample_rate, root, loop=None, correction=0):
        """pcm16: lista de enteros. Devuelve el índice del sample."""
        start = len(self.smpl) // 2
        if isinstance(pcm16, (bytes, bytearray)):
            self.smpl += pcm16
            count = len(pcm16) // 2
        else:
            self.smpl += struct.pack('<%dh' % len(pcm16), *pcm16)
            count = len(pcm16)
        self.smpl += b'\x00\x00' * 46  # relleno obligatorio
        end = start + count
        ls, le = loop if loop else (start, end)
        if loop:
            ls, le = start + loop[0], start + loop[1]
        self.rec['shdr'].append([name.encode()[:20], start, end, ls, le, sample_rate, root, correction, 0, 1])
        return len(self.rec['shdr']) - 1

    def add_instrument(self, name, zones):
        """zones: lista de listas [(oper, amount_u16)] (sampleID al final)."""
        self.rec['inst'].append([name.encode()[:20], len(self.rec['ibag'])])
        for z in zones:
            self.rec['ibag'].append([len(self.rec['igen']), len(self.rec['imod'])])
            for g in order_gens(z, 'sampleID'):
                self.rec['igen'].append(list(g))
        return len(self.rec['inst']) - 1

    def add_preset(self, name, bank, program, zones):
        self.rec['phdr'].append([name.encode()[:20], program, bank, len(self.rec['pbag']), 0, 0, 0])
        for z in zones:
            self.rec['pbag'].append([len(self.rec['pgen']), len(self.rec['pmod'])])
            for g in order_gens(z, 'instrument'):
                self.rec['pgen'].append(list(g))

    # ---------- escribir ----------
    def save(self, path):
        t = self.term
        phdr = self.rec['phdr'] + [[b'EOP', 0, 0, len(self.rec['pbag']), 0, 0, 0]]
        pbag = self.rec['pbag'] + [[len(self.rec['pgen']), len(self.rec['pmod'])]]
        inst = self.rec['inst'] + [[b'EOI', len(self.rec['ibag'])]]
        ibag = self.rec['ibag'] + [[len(self.rec['igen']), len(self.rec['imod'])]]
        shdr = self.rec['shdr'] + [[b'EOS', 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        lists = dict(phdr=phdr, pbag=pbag, pmod=self.rec['pmod'] + [t['pmod']], pgen=self.rec['pgen'] + [t['pgen']],
                     inst=inst, ibag=ibag, imod=self.rec['imod'] + [t['imod']], igen=self.rec['igen'] + [t['igen']], shdr=shdr)

        def chunk(cid, payload):
            out = cid.encode() + struct.pack('<I', len(payload)) + payload
            return out + (b'\x00' if len(payload) & 1 else b'')

        pdta = b'pdta'
        for k in ('phdr', 'pbag', 'pmod', 'pgen', 'inst', 'ibag', 'imod', 'igen', 'shdr'):
            f, _ = self.fmt[k]
            pdta += chunk(k, b''.join(struct.pack(f, *r) for r in lists[k]))
        sdta = b'sdta' + chunk('smpl', bytes(self.smpl))
        if self.sm24 is not None:
            sm24 = self.sm24 + b'\x00' * (len(self.smpl) // 2 - len(self.sm24))
            sdta += chunk('sm24', sm24)
        body = b'sfbk' + chunk('LIST', b'INFO' + self.info) + chunk('LIST', sdta) + chunk('LIST', pdta)
        open(path, 'wb').write(b'RIFF' + struct.pack('<I', len(body)) + body)


def u16(v):
    return v & 0xFFFF


def rng(lo, hi):
    return lo | (hi << 8)


def order_gens(z, last):
    """keyRange y velRange primero; instrument/sampleID al final (regla del SF2)."""
    lastnum = GEN[last]
    kr = [g for g in z if g[0] == 43]
    vr = [g for g in z if g[0] == 44]
    mid = [g for g in z if g[0] not in (43, 44, lastnum)]
    end = [g for g in z if g[0] == lastnum]
    return kr + vr + mid + end
