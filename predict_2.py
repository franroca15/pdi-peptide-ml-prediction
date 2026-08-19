#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import joblib
import torch
import esm
import os 
from typing import List

# --- NEW FUNCTION: CONTACT LOGIC ---
def estimate_physical_contacts(seq: str, score: float) -> float:
    """
    Estimates the number of possible contacts based on the biochemical 
    propensity of the interface and the model's confidence.
    """
    # Contact propensity scale (frequency in PPI interfaces)
    contact_scale = {
        'W': 1.25, 'Y': 1.15, 'R': 1.10, 'C': 1.10, 'H': 0.95,
        'F': 0.90, 'M': 0.85, 'K': 0.85, 'L': 0.80, 'I': 0.80,
        'V': 0.75, 'D': 0.70, 'E': 0.70, 'T': 0.65, 'S': 0.65,
        'N': 0.60, 'Q': 0.60, 'A': 0.55, 'G': 0.45, 'P': 0.40
    }
    
    # Base sum according to the sequence composition
    base_contacts = sum([contact_scale.get(aa.upper(), 0.6) for aa in seq])
    
    # Adjustment factor based on predicted affinity (score)
    # The higher the score, the higher the probability of a "closed" and stable interface.
    total = base_contacts * (1 + (score ** 2))
    return round(total, 1)

def info(msg: str):
    print(f"[INFO] {msg}")

def load_esm_model(model_name="esm2_t6_8M_UR50D"):
    if not hasattr(esm, "pretrained"):
        raise RuntimeError("Please install 'fair-esm'.")
    get_model = getattr(esm.pretrained, model_name)
    model, alphabet = get_model()
    model.eval()
    return model, alphabet

def embed_sequences_esm(model, alphabet, seqs: List[str], batch_size: int = 64, device: str = "cpu") -> np.ndarray:
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i+batch_size]
            data = [(str(j), s) for j, s in enumerate(batch_seqs)]
            _, strs, tokens = batch_converter(data)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[model.num_layers])
            reps = out["representations"][model.num_layers]
            for rep, seq in zip(reps, strs):
                L = len(seq)
                vec = rep[1:L+1].mean(0)
                all_vecs.append(vec.cpu().numpy())
    return np.vstack(all_vecs)

def find_col(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns: return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf-path", required=True)
    ap.add_argument("--new-data", required=True)
    ap.add_argument("--model-name", default="esm2_t6_8M_UR50D")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="ppi_prediction_final_esm.csv")
    args = ap.parse_args()
    device = torch.device(args.device)

    # 1. Load data and models
    clf = joblib.load(args.clf_path)
    model, alphabet = load_esm_model(args.model_name)
    df_new = pd.read_csv(args.new_data)
    
    pep_col = find_col(df_new, ["pep_sequence", "peptide_sequence", "peptide"])
    prot_col = find_col(df_new, ["prot_sequence", "protein_sequence", "protein"])
    
    # 2. Embeddings
    pep_seqs = df_new[pep_col].astype(str).unique()
    prot_seqs = df_new[prot_col].astype(str).unique()
    pep_vecs = embed_sequences_esm(model, alphabet, pep_seqs, batch_size=args.batch_size, device=device)
    prot_vecs = embed_sequences_esm(model, alphabet, prot_seqs, batch_size=args.batch_size, device=device)
    pep_dict = {s: v for s, v in zip(pep_seqs, pep_vecs)}
    prot_dict = {s: v for s, v in zip(prot_seqs, prot_vecs)}

    # 3. Concatenation and Prediction
    X_list = [np.concatenate([pep_dict[row[pep_col]], prot_dict[row[prot_col]]]) for _, row in df_new.iterrows()]
    X_new = np.vstack(X_list)
    y_scores = clf.predict_proba(X_new)[:, 1]

    # --- 4. INTEGRATION OF RESULTS WITH CONTACTS ---
    info("Adding possible contacts column...")
    df_new['prediction_score'] = y_scores
    df_new['prediction_class'] = (y_scores >= 0.5).astype(int)
    
    # Apply the physical estimation function
    df_new['num_contactos_posibles'] = df_new.apply(
        lambda x: estimate_physical_contacts(str(x[pep_col]), x['prediction_score']), axis=1
    )
    
    # 5. Save results
    df_new.to_csv(args.out, index=False)
    info(f"Process finished. Results saved in: {args.out}")

if __name__ == "__main__":
    main()
