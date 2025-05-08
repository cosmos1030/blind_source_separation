#!/usr/bin/env python3
"""
Hu & Wang (2002) 단일 마이크 CASA (8 kHz 기준) – 중간 단계 시각화 포함
  • 입력  : --wav  <mono wav>
  • 출력  : --out  <dir>  (없으면 자동 생성)
  ├── gt_spec.png          : Gammatone 스펙트로그램
  ├── f0_track.png         : 글로벌 F0 곡선
  ├── mask.png             : 최종 T‑F 마스크(썸네일)
  ├── foreground.wav
  └── background.wav
"""
import os, argparse, numpy as np, matplotlib.pyplot as plt
import librosa, soundfile as sf
from gammatone.gtgram import gtgram
from scipy.signal import hilbert, butter, lfilter, medfilt
from scipy.fft import rfft, irfft

# ───────── 파라미터 (필요하면 --sr 로 바꿔도 됨) ─────────
SR               = 8000
WIN_T_GT, HOP_T_GT = 0.025, 0.010
N_CH, FMIN       = 128, 50
WIN_T_ACF, HOP_T_ACF = 0.032, 0.010
F0_MIN, F0_MAX   = 50, 500
VOIC_TH          = 0.30
SEG_MIN_FR       = int(0.05 / HOP_T_GT)  # 50 ms

# ───────── 유틸 ─────────
def frame(sig, L, H):
    n = 1 + (len(sig)-L)//H
    return np.stack([sig[i*H:i*H+L] for i in range(n)])

def save_png(path, fig):
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

# ───────── 1. 로드 & GT 스펙트로그램 ─────────
def load_and_gt(wav):
    x, _ = librosa.load(wav, sr=SR, mono=True)
    gt = gtgram(x, SR, WIN_T_GT, HOP_T_GT, N_CH, FMIN)   # (B,T)
    # 시각화
    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(gt[::-1], aspect='auto', origin='lower',
                   extent=[0, len(x)/SR, FMIN, SR/2])
    ax.set_xlabel('Time [s]'); ax.set_ylabel('Frequency [Hz]')
    ax.set_title('Gammatone Spectrogram'); fig.colorbar(im, ax=ax)
    return x, gt, fig

# ───────── 2. 연속처리 (envelope / ACF / F0) ─────────
def envelopes(gt): return np.abs(hilbert(np.maximum(gt,0), axis=1))

def band_acf(env):
    L, H = int(SR*WIN_T_ACF), int(SR*HOP_T_ACF)
    nF = 1 + (env.shape[1]-L)//H
    acf = np.zeros((N_CH, nF, L))
    win = np.hanning(L)
    for b in range(N_CH):
        frames = frame(env[b], L, H) * win
        spec   = np.abs(rfft(frames, axis=1))**2
        acf[b] = irfft(spec, n=L, axis=1)
    return acf                                            # (B,F,L)

def global_f0(acf, env):
    w = env.mean(1, keepdims=True); w /= w.sum()
    gacf = (acf * w[:,:,None]).sum(0)                     # (F,L)
    lag_min, lag_max = int(SR/F0_MAX), int(SR/F0_MIN)
    lag_max = min(lag_max, gacf.shape[1]-1)
    if lag_max <= lag_min: return np.zeros(gacf.shape[0])
    idx = lag_min + np.argmax(gacf[:, lag_min:lag_max], axis=1)
    norm = gacf[np.arange(len(idx)), idx] / gacf[:,0]
    f0 = np.where(norm >= VOIC_TH, SR/idx, 0.)
    return medfilt(f0, 3)

def plot_f0(f0, out_png):
    t = np.arange(len(f0))*HOP_T_ACF
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(t, f0); ax.set_ylim(0, F0_MAX+50)
    ax.set_xlabel('Time [s]'); ax.set_ylabel('F0 [Hz]')
    ax.set_title('Global F0 Track')
    save_png(out_png, fig)

# ───────── 3. T‑F 라벨 ─────────
def label_tf(acf, env, f0):
    B,F,L = acf.shape; mask = np.zeros((B,F), bool)
    freqs = np.linspace(FMIN, SR/2, N_CH)
    res = np.where(freqs<=1000)[0]

    for b in res:
        for f in range(F):
            if f0[f]==0: continue
            lag=int(SR/f0[f])
            if lag<L and acf[b,f,lag]/acf[b,f,0]>.95: mask[b,f]=1

    bp_b,bp_a = butter(2,[80/(SR/2),500/(SR/2)],'band')
    for b in range(res[-1]+1, B):
        am = lfilter(bp_b,bp_a,env[b])
        Lf, H = int(SR*WIN_T_ACF), int(SR*HOP_T_ACF)
        for f in range(F):
            if f0[f]==0: continue
            seg = am[f*H:f*H+Lf]; seg-=seg.mean()
            if len(seg)<Lf: break
            freq = np.fft.rfftfreq(Lf, 1/SR)[np.argmax(np.abs(rfft(seg)))]
            if abs(freq - f0[f]) / (f0[f]+1e-6) < .12: mask[b,f]=1
    return mask

# def post_mask(mask, env):
#     # 50 ms 미만 FG 제거
#     for b in range(N_CH):
#         pos = np.where(mask[b])[0]
#         segs = np.split(pos, np.where(np.diff(pos)>1)[0]+1)
#         for s in segs:
#             if len(s)<SEG_MIN_FR: mask[b,s]=0
#     undec = ~mask
#     th = np.percentile(env,75)
#     mask |= undec & (env>th)
#     return mask.astype(float)

def post_mask(mask, env):
    """mask: (B,nF)  env: (B,T_gt) → env_fr: (B,nF) 로 맞춘 뒤 사용"""
    # ➊ env 를 ACF 창·홉으로 프레이밍해 밴드별 프레임 에너지 구함
    L, H = int(SR*WIN_T_ACF), int(SR*HOP_T_ACF)
    nF   = mask.shape[1]
    env_fr = np.zeros_like(mask, dtype=float)   # (B,nF)
    for b in range(N_CH):
        for f in range(nF):
            seg = env[b, f*H : f*H+L]
            env_fr[b, f] = seg.mean()

    # ➋ 50 ms 미만 FG 제거
    for b in range(N_CH):
        pos = np.where(mask[b])[0]
        segs = np.split(pos, np.where(np.diff(pos)>1)[0]+1)
        for s in segs:
            if len(s) < SEG_MIN_FR:
                mask[b, s] = 0

    # ➌ undecided 영역을 에너지 기준으로 채움
    undec = ~mask
    th = np.percentile(env_fr, 75)
    mask |= undec & (env_fr > th)
    return mask.astype(float)


def save_mask_png(mask, out_png):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(mask[::-1], aspect='auto', origin='lower',
              cmap='gray_r', interpolation='nearest')
    ax.set_xlabel('Frame'); ax.set_ylabel('Channel')
    ax.set_title('Binary T‑F Mask'); save_png(out_png, fig)

# ───────── 4. OLA 복원 ─────────
def overlap_add(x, mask):
    hop = int(SR*HOP_T_GT); win = np.hanning(int(SR*WIN_T_GT))
    y_t, y_b = np.zeros_like(x), np.zeros_like(x)
    for f in range(mask.shape[1]):
        g = mask[:,f].mean()
        i=f*hop; j=i+len(win)
        if j>len(x): break
        y_t[i:j]+=g*x[i:j]*win
        y_b[i:j]+=(1-g)*x[i:j]*win
    return y_t, y_b

# ───────── MAIN ─────────
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', required=True, help='mono wav 파일')
    ap.add_argument('--out', default='./out', help='출력 폴더')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) GT 스펙트로그램
    x, gt, fig_gt = load_and_gt(args.wav)
    save_png(os.path.join(args.out, 'gt_spec.png'), fig_gt)

    # 2) Envelope / ACF / F0
    env  = envelopes(gt)
    acf  = band_acf(env)
    f0   = global_f0(acf, env)
    plot_f0(f0, os.path.join(args.out, 'f0_track.png'))

    # 3) 마스크
    m0   = label_tf(acf, env, f0)
    mask = post_mask(m0, env)
    save_mask_png(mask, os.path.join(args.out, 'mask.png'))

    # 4) 복원 & 저장
    y_fg, y_bg = overlap_add(x, mask)
    sf.write(os.path.join(args.out, 'foreground.wav'),  y_fg, SR)
    sf.write(os.path.join(args.out, 'background.wav'), y_bg, SR)

    print('✔ Done. 결과 →', args.out)
