#!/usr/bin/env python3
"""
V8: HRTF × Random-Forest — Implementação correta com mapeamento adequado

Baseado na documentação HUTUBS:
- x1-x17: medidas do corpo
- d1-d10: medidas da orelha (L_ para esquerda, R_ para direita)
- theta1-theta2: ângulos da orelha
- 5 áreas calculadas segundo o paper
"""
import os, re, glob, numpy as np, pandas as pd, netCDF4
from sklearn.ensemble import RandomForestRegressor
import json

np.random.seed(42)
root = os.path.dirname(__file__)
data_dir = os.path.join(root, "data", "hutubs")

# ---------------------------------------------------------------- CSV
df = pd.read_csv(os.path.join(data_dir, "AntrhopometricMeasures.csv"))
df.columns = [c.strip() for c in df.columns]
df["SubjectID"] = df["SubjectID"].astype(int)

# Verificar quais colunas temos
print("Colunas disponíveis no CSV:")
print(df.columns.tolist())

# Para o paper, precisamos mapear para a1-a19
# a1-a14: algumas das medidas x e d
# a15-a19: áreas calculadas

# Criar mapeamento segundo o paper (Tabela I):
# a1-a5: x1-x5 (head/pinna measurements)
# a6: d1 (cavum concha height)
# a7: d2 (cymba concha height)
# a8: d3 (cavum concha width)
# a9: d4 (fossa height)
# a10: d5 (pinna height)
# a11: d6 (pinna width)
# a12: d7 (intertragal incisure width)
# a13: theta1 (pinna rotation angle)
# a14: theta2 (pinna flare angle)

# Como temos medidas separadas para L/R, vamos criar datasets separados
def create_features_for_ear(df, ear='L'):
    """Cria features a1-a19 para uma orelha específica"""
    features = pd.DataFrame()
    features['SubjectID'] = df['SubjectID']
    
    # a1-a5: medidas do corpo (iguais para ambas orelhas)
    features['a1'] = df['x1']  # head width
    features['a2'] = df['x2']  # head height
    features['a3'] = df['x3']  # head depth
    features['a4'] = df['x4']  # pinna offset down
    features['a5'] = df['x5']  # pinna offset back
    
    # a6-a14: medidas específicas da orelha
    features['a6'] = df[f'{ear}_d1']   # cavum concha height
    features['a7'] = df[f'{ear}_d2']   # cymba concha height
    features['a8'] = df[f'{ear}_d3']   # cavum concha width
    features['a9'] = df[f'{ear}_d4']   # fossa height
    features['a10'] = df[f'{ear}_d5']  # pinna height
    features['a11'] = df[f'{ear}_d6']  # pinna width
    features['a12'] = df[f'{ear}_d7']  # intertragal incisure width
    features['a13'] = df[f'{ear}_theta1']  # pinna rotation angle
    features['a14'] = df[f'{ear}_theta2']  # pinna flare angle
    
    # a15-a19: áreas calculadas (segundo o paper)
    features['a15'] = features['a6'] * features['a8'] / 2  # cavum concha area
    features['a16'] = features['a7'] * features['a8'] / 2  # cymba concha area
    features['a17'] = features['a9'] * features['a11'] / 2  # fossa area
    features['a18'] = features['a10'] * features['a11'] / 2  # pinna area
    features['a19'] = features['a12'] * (features['a6'] + features['a8']) / 2  # intertragal area
    
    return features

# Criar datasets para cada orelha
df_left = create_features_for_ear(df, 'L')
df_right = create_features_for_ear(df, 'R')

# Remover linhas com NaN
df_left = df_left.dropna()
df_right = df_right.dropna()

print(f"\nSujeitos com dados completos:")
print(f"  Orelha esquerda: {len(df_left)}")
print(f"  Orelha direita: {len(df_right)}")

# ------------------------------------------------------------- constantes
fs = 44100
tgt_freqs = np.linspace(1000, 12000, 64, dtype=np.float32)
pos_tgt = [(0,0), (40,0), (320,0), (0,30), (0,-30)]  # 5 posições

# ------------------------------------------------------------ ler SOFA
cache = {}
for fp in glob.glob(os.path.join(data_dir, "pp*_HRIRs_measured.sofa")):
    m = re.search(r"pp(\d+)_", fp)
    if not m:
        continue
    sid = int(m.group(1))
    
    # Verificar se temos dados antropométricos para este sujeito
    if sid not in df_left.set_index('SubjectID').index:
        continue
    
    try:
        ds = netCDF4.Dataset(fp)
        if "Data.IR" not in ds.variables or "SourcePosition" not in ds.variables:
            ds.close()
            continue
        
        ir = ds["Data.IR"][:]  # (pos,2,512)
        pos = ds["SourcePosition"][:, :2]  # (pos,2)
        ds.close()
        
        if ir.shape[0] < 440:  # arquivo incompleto
            continue
        
        orig_f = np.fft.rfftfreq(ir.shape[-1], 1/fs)
        mask = (orig_f >= 1000) & (orig_f <= 12000)
        band_f = orig_f[mask]
        idx_map = [np.argmin(np.sum((pos - p)**2, axis=1)) for p in pos_tgt]
        
        for ear in (0,1):
            mag = np.abs(np.fft.rfft(ir[:, ear, :], axis=-1)) + 1e-10
            mag_db = 20*np.log10(mag[:, mask])
            for k,pidx in enumerate(idx_map):
                cache[(sid, ear, k)] = np.interp(
                    tgt_freqs, band_f, mag_db[pidx]).astype(np.float32)
    except Exception as e:
        print(f"Erro ao processar {fp}: {e}")
        continue

valid_sids = sorted({sid for (sid,_,_) in cache})
print(f"\n▶ Sujeitos com SOFA + antropometria: {len(valid_sids)}")

if len(valid_sids) < 20:
    raise RuntimeError("Precisamos de ≥20 sujeitos.")

# Excluir sujeitos problemáticos identificados
exclude = [18, 56, 79, 80, 92, 94]
valid_sids = [s for s in valid_sids if s not in exclude]
print(f"▶ Após exclusões: {len(valid_sids)} sujeitos")

valid_sids = np.array(valid_sids, dtype=int)
np.random.shuffle(valid_sids)
test_sub = valid_sids[:10]
train_sub = valid_sids[10:]

print(f"▶ Treino: {len(train_sub)} | Teste: {len(test_sub)}")

# ----------------------------------------------------------- utilidades
def build(sub_ids, ear, pos_id):
    X, Y = [], []
    # Escolher dataset correto
    df_ear = df_left if ear == 0 else df_right
    df_ear = df_ear.set_index('SubjectID')
    
    for sid in sub_ids:
        key = (sid, ear, pos_id)
        if key not in cache or sid not in df_ear.index:
            continue
        
        spec = cache[key]
        # Usar apenas a1-a19 (19 features antropométricas)
        feats = df_ear.loc[sid, ['a'+str(i) for i in range(1,20)]].to_numpy(dtype=np.float32)
        rep = np.repeat(feats.reshape(1,-1), 64, axis=0)
        X.append(np.hstack([rep, tgt_freqs.reshape(-1,1)]))
        Y.append(spec)
    
    if not X:
        return None, None
    return np.vstack(X), np.hstack(Y)

def r2_corr(y, yhat):
    r = np.corrcoef(y, yhat)[0,1]
    return (r*r) if not np.isnan(r) else 0.0

# Parâmetros do modelo segundo o paper
max_feat = 18  # "18 selected variables" - sem contar frequência
R2, SD = {0:[],1:[]}, {0:[],1:[]}

# ----------------------------------------------------------- treino/teste
print("\nTreinando modelos...")
for ear in (0,1):
    for pid in range(5):
        X_tr, y_tr = build(train_sub, ear, pid)
        if X_tr is None:
            R2[ear].append(0.0)
            SD[ear].append(np.nan)
            continue
        
        print(f"  Orelha {'esquerda' if ear==0 else 'direita'}, posição {pos_tgt[pid]}")
        print(f"    Dados treino: {X_tr.shape}")
        
        rf = RandomForestRegressor(
            n_estimators=500,
            max_features=max_feat,  
            min_samples_split=2,    
            min_samples_leaf=5,     
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42)
        rf.fit(X_tr, y_tr)
        R2[ear].append(r2_corr(y_tr, rf.oob_prediction_))
        
        X_te, y_te = build(test_sub, ear, pid)
        if X_te is not None:
            # Cálculo correto de SD: RMS das diferenças em dB
            y_pred = rf.predict(X_te)
            sd_value = np.sqrt(np.mean((y_pred - y_te)**2))  # RMS
            SD[ear].append(sd_value)
        else:
            SD[ear].append(np.nan)

# ------------------------------------------------------------- saída
hdr = ["Ear","(0°,0°)","(40°,0°)","(320°,0°)","(0°,30°)","(0°,-30°)","Mean"]

print("\n" + "="*80)
print("TABLE II – Determination Coefficients R²")
print("="*80)
print(" | ".join(hdr))
for ear,label in ((0,"Left"),(1,"Right")):
    row = [f"{v*100:5.1f}%" for v in R2[ear]]
    print(" | ".join([label]+row+[f"{np.nanmean(R2[ear])*100:5.1f}%"]))

print("\n" + "="*80)
print("Spectral Distortion SD (dB)")
print("="*80)
print(" | ".join(hdr))
for ear,label in ((0,"Left"),(1,"Right")):
    row = [f"{v:5.2f}" for v in SD[ear]]
    print(" | ".join([label]+row+[f"{np.nanmean(SD[ear]):5.2f}"]))

# Salvar resultados
results = {
    "method": "V8 - Correct parameter mapping (a1-a19)",
    "features_used": "a1-a19 according to paper Table I",
    "model_params": {
        "n_estimators": 500,
        "max_features": 18,
        "min_samples_split": 2,
        "min_samples_leaf": 5
    },
    "results": {
        "R2_left": [float(v) for v in R2[0]],
        "R2_right": [float(v) for v in R2[1]],
        "R2_mean_left": float(np.nanmean(R2[0])),
        "R2_mean_right": float(np.nanmean(R2[1])),
        "SD_left": [float(v) for v in SD[0]],
        "SD_right": [float(v) for v in SD[1]],
        "SD_mean_left": float(np.nanmean(SD[0])),
        "SD_mean_right": float(np.nanmean(SD[1]))
    }
}

os.makedirs('optimization_results', exist_ok=True)
with open('optimization_results/v8_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n▶ Resultados salvos em optimization_results/v8_results.json")
print("\nComparação com o paper (Tabela II):")
print("- Paper: R² ~90.1% (left), ~91.1% (right)")
print(f"- V8: R² ~{np.nanmean(R2[0])*100:.1f}% (left), ~{np.nanmean(R2[1])*100:.1f}% (right)")
print("- Paper: SD ~4.74 dB")
print(f"- V8: SD ~{np.mean([np.nanmean(SD[0]), np.nanmean(SD[1])]):.2f} dB")