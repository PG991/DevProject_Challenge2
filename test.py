from dataset.dataset_ESC50 import ESC50
import config

# Hier erzeugst du dein Dataset nur für Fold 1 (zur Kontrolle)
ds = ESC50(
    root=config.esc50_path,
    subset="train",
    test_folds={1},
    global_mean_std=(0.0, 1.0),
    download=False
)

# Hole das erste Beispiel
fn, feat, cls = ds[0]

print("Dateiname:", fn)
print("Label:", cls)
print("Feature-Shape:", feat.shape)
# Erwartet z. B. torch.Size([1, 64, T]), 
#   wobei T ≃ 861 (bei hop_length=256 und 5 Sekunden Audio)
