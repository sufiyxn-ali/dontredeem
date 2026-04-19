import os
import yaml
from huggingface_hub import snapshot_download, login

def main():
    print("="*60)
    print("   Pyannote Offline Deployment - Localizer Tool")
    print("="*60)
    print("\nThis script will download Pyannote Models locally exactly once.")
    print("You need your Hugging Face Token (starts with 'hf_') and must have")
    print("agreed to the conditions for both models on their Hugging Face pages:\n")
    print("1. pyannote/speaker-diarization-3.1")
    print("2. pyannote/segmentation-3.0\n")
    
    try:
        # Prompt user directly (no hardcoded tokens)
        token = input("Enter your Hugging Face Token: ").strip()
        if not token:
            print("No token provided. Exiting.")
            return
            
        print("\n[!] Authenticating...")
        login(token)
        
        # Determine strict paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        base_dir = os.path.join(project_root, 'models', 'pyannote')
        os.makedirs(base_dir, exist_ok=True)
        
        repos = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/segmentation-3.0",
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        ]
        
        dirs = {}
        # Fetch snapshots directly to disk avoiding the cache reliance
        for repo in repos:
            print(f"\n[+] Downloading {repo}...")
            repo_dir = os.path.join(base_dir, repo.split('/')[-1])
            # By setting local_dir, we physically drop the files there
            # local_dir_use_symlinks=False ensures they are pure files, not pointers
            snapshot_download(repo_id=repo, local_dir=repo_dir, local_dir_use_symlinks=False, token=token)
            dirs[repo] = repo_dir
            
        # Core Diarization Pipeline YAML needs patching
        main_config_path = os.path.join(dirs["pyannote/speaker-diarization-3.1"], "config.yaml")
        
        print("\n[!] Patching config.yaml for local offline execution...")
        with open(main_config_path, 'r') as f:
            config_content = yaml.safe_load(f)
            
        # We explicitly supply the absolute paths to the downloaded model checkpoint weights files (.bin)
        config_content['pipeline']['params']['segmentation'] = os.path.join(dirs["pyannote/segmentation-3.0"], "pytorch_model.bin")
        config_content['pipeline']['params']['embedding'] = os.path.join(dirs["pyannote/wespeaker-voxceleb-resnet34-LM"], "pytorch_model.bin")
        
        with open(main_config_path, 'w') as f:
            yaml.dump(config_content, f)
            
        print(f"\n" + "="*60)
        print(f"SUCCESS: Pyannote Diarization is now 100% Offline and Localized.")
        print(f"Main Config File: {main_config_path}")
        print("Your pipeline will now run entirely without the Hugging Face API.")
        print("="*60)
        
    except Exception as e:
        print(f"\n[X] Error during localization: {e}")
        print("Check if your token is correct and that you agreed to the conditions!")

if __name__ == "__main__":
    main()
