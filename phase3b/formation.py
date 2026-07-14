import glob, os, pandas as pd
from phase3b.gt_labels import formation_label

def report(split="valid", gt_root="data/SoccerNetGS"):
    rows = []
    for jp in sorted(glob.glob(f"{gt_root}/{split}/SNGS-*/Labels-GameState.json")):
        seq = os.path.basename(os.path.dirname(jp)).replace("SNGS-", "")
        for team, f in formation_label(jp).items():
            rows.append((seq, team, f))
    df = pd.DataFrame(rows, columns=["seq", "team", "formation"])
    print("GT formation distribution (best-effort, partial-observability limited):")
    print(df.formation.value_counts())
    print("\nNOTE: only ~5-6 players/team visible per frame; clip-aggregated GT still")
    print("under-observes outfield lines, so these labels are noisy. Reported as a")
    print("limitation, not a reliable classification target.")
    df.to_csv(f"phase3b/formation_{split}.csv", index=False)
    return df

if __name__ == "__main__":
    report()