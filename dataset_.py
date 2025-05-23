from dataset.dataset_ESC50 import ESC50
import config

testset = ESC50(subset="test", root=config.esc50_path, download=True)
print(f"Anzahl Test-Samples: {len(testset)}")
