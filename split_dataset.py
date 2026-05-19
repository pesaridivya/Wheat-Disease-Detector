import os, shutil, random

src = r"C:\Users\pesar\OneDrive\Desktop\ai\Wheat_Disease\train"
dst = r"C:\Users\pesar\OneDrive\Desktop\ai\Wheat_Small"
random.seed(42)

classes = ["Black Rust", "Brown Rust", "Healthy Wheat", "Yellow Rust"]

for cls in classes:
    in_dir = os.path.join(src, cls)
    imgs   = os.listdir(in_dir)
    random.shuffle(imgs)
    splits = {"train": imgs[:800], "val": imgs[800:1000]}
    for split, selected in splits.items():
        out_dir = os.path.join(dst, split, cls)
        os.makedirs(out_dir, exist_ok=True)
        for img in selected:
            shutil.copy2(os.path.join(in_dir, img),
                         os.path.join(out_dir, img))
        print(f"  {split}/{cls}: {len(selected)} images")

print("\nDone!")