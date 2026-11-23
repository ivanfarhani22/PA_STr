import numpy as np

def readiness_pct(y_hat, cot, resid, mode="rel"):
    """Persentase residual historis yang membuat estimasi akhir <= COT.
    mode 'rel': y = y_hat * (1 + e_rel)   | mode 'abs': y = y_hat + e"""
    resid = np.asarray(resid)
    est = y_hat * (1 + resid) if mode == "rel" else y_hat + resid
    return 100.0 * np.mean(est <= cot)

def category(pct):
    if pct >= 90: return "Sangat Siap"
    if pct >= 75: return "Siap"
    if pct >= 50: return "Cukup Siap"
    if pct >= 25: return "Kurang Siap"
    return "Belum Siap"