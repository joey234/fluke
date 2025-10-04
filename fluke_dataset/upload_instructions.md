# FLUKE Dataset Upload Instructions

## Prerequisites

1. Install Hugging Face CLI:
```bash
pip install huggingface_hub
```

2. Login with your token:
```bash
huggingface-cli login
```
Enter your token when prompted.

## Method 1: Using Hugging Face CLI (Recommended)

### Step 1: Create the repository
```bash
huggingface-cli repo create fluke --type dataset
```

### Step 2: Clone the repository
```bash
git clone https://huggingface.co/datasets/YOUR_USERNAME/fluke
cd fluke
```

### Step 3: Copy your organized files
```bash
# From your fluke directory, copy the organized dataset
cp -r fluke_dataset/* ./
```

### Step 4: Upload to Hugging Face
```bash
git add .
git commit -m "Add FLUKE dataset"
git push
```

## Method 2: Using Python API

```python
from huggingface_hub import HfApi, create_repo
import os

# Initialize API (make sure you're logged in with huggingface-cli login)
api = HfApi()

# Create repository
repo_id = "YOUR_USERNAME/fluke"
create_repo(repo_id, repo_type="dataset", private=False)

# Upload all files
api.upload_folder(
    folder_path="./fluke_dataset",
    repo_id=repo_id,
    repo_type="dataset"
)
```

## Method 3: Direct Web Upload

1. Go to https://huggingface.co/new-dataset
2. Create a new dataset called "fluke"
3. Upload files through the web interface
4. Make sure to upload:
   - README.md (as the main dataset card)
   - All files from fluke_dataset/

## Verification

After upload, users should be able to load your dataset with:

```python
from datasets import load_dataset

# Load all tasks
dataset = load_dataset("YOUR_USERNAME/fluke", "all")

# Load specific task
sa_data = load_dataset("YOUR_USERNAME/fluke", "sa")
```

## Important Notes

- Replace `YOUR_USERNAME` with your actual Hugging Face username
- The dataset will be public by default
- Make sure all files from `fluke_dataset/` are uploaded
- The README.md serves as your dataset card on Hugging Face

## Troubleshooting

If you encounter issues:
1. Make sure you're logged in: `huggingface-cli whoami`
2. Check your token has write permissions
3. Verify file paths are correct
4. Try the Python API method if CLI doesn't work 