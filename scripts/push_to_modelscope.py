"""
Push DPO LoRA adapter + model card to ModelScope Hub.
Simplified version — skip whoami check (just try and see).
"""
import time
from pathlib import Path
from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

MODELSCOPE_USERNAME = "linglcn"
REPO_NAME = "Psy-Qwen-DPO-LoRA"
REPO_ID = f"{MODELSCOPE_USERNAME}/{REPO_NAME}"
UPLOAD_DIR = Path("/home/claude/hf_upload")
COMMIT_MESSAGE = "Initial release: DPO LoRA adapter (74.88% win rate vs SFT)"


def main():
    api = HubApi()

    # 1. Create repo (if not exists)
    print(f"[Step 1/2] Creating repo {REPO_ID} ...")
    try:
        api.create_model(
            model_id=REPO_ID,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.MIT,
            chinese_name="心理咨询 DPO LoRA",
        )
        print(f"  ✅ Created")
    except Exception as e:
        msg = str(e).lower()
        if "exist" in msg or "已存在" in str(e):
            print(f"  ℹ️  Repo already exists, will overwrite files")
        else:
            print(f"  ⚠️  create_model error: {e}")
            print(f"  Continuing to upload anyway...")

    # 2. Upload files
    print(f"\n[Step 2/2] Uploading files from {UPLOAD_DIR} ...")
    files = sorted(UPLOAD_DIR.iterdir())
    total = len(files)
    t_start = time.time()

    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  [{i}/{total}] {f.name} ({size_mb:.2f} MB) ...", end=" ", flush=True)
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=REPO_ID,
                commit_message=COMMIT_MESSAGE,
            )
            print(f"✅ {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"❌")
            print(f"      Error: {e}")

    elapsed = time.time() - t_start
    print(f"\n✅ Done in {elapsed:.0f}s")
    print(f"   View at: https://modelscope.cn/models/{REPO_ID}")


if __name__ == "__main__":
    main()