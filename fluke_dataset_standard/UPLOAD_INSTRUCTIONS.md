# FLUKE Dataset Upload Instructions (Standard Format)

## ✅ **Problem Solved!**

The "arbitrary Python code execution" error has been fixed by converting the dataset to standard Parquet format. Your dataset will now work with Hugging Face's automatic dataset viewer!

## 📁 **What Changed**

- ❌ Removed: Custom loading script (`fluke.py`)
- ❌ Removed: Complex JSON directory structure  
- ✅ Added: Standard Parquet files for each task
- ✅ Added: Combined `train.parquet` with all tasks
- ✅ Enabled: Automatic Hugging Face dataset viewer

## 🚀 **Upload Steps**

### Prerequisites

```bash
pip install huggingface_hub datasets
huggingface-cli login  # Enter your token
```

### Method 1: Python API (Recommended)

```python
from huggingface_hub import HfApi, create_repo

# Initialize API
api = HfApi()

# Create repository (replace YOUR_USERNAME)
repo_id = "YOUR_USERNAME/fluke"
create_repo(repo_id, repo_type="dataset", private=False)

# Upload the standard format dataset
api.upload_folder(
    folder_path="./fluke_dataset_standard",
    repo_id=repo_id,
    repo_type="dataset"
)

print(f"✅ Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
```

### Method 2: Git CLI

```bash
# Create repository
huggingface-cli repo create fluke --type dataset

# Clone and upload
git clone https://huggingface.co/datasets/YOUR_USERNAME/fluke
cd fluke
cp -r ../fluke_dataset_standard/* ./
git add .
git commit -m "Add FLUKE dataset in standard format"
git push
```

## 📊 **Usage After Upload**

Your dataset will be automatically detected and users can load it with:

```python
from datasets import load_dataset

# Load all tasks combined
dataset = load_dataset("YOUR_USERNAME/fluke")

# Load specific task
coref_data = load_dataset("YOUR_USERNAME/fluke", data_files="coref.parquet")
ner_data = load_dataset("YOUR_USERNAME/fluke", data_files="ner.parquet")
sa_data = load_dataset("YOUR_USERNAME/fluke", data_files="sa.parquet")
dialogue_data = load_dataset("YOUR_USERNAME/fluke", data_files="dialogue.parquet")

# Filter by modification type
train_data = dataset["train"]
negation_examples = train_data.filter(lambda x: x["modification_type"] == "negation")
```

## ✨ **Benefits of Standard Format**

1. **✅ Dataset Viewer Enabled**: Users can browse your data on Hugging Face
2. **✅ Faster Loading**: Parquet format is optimized for ML workloads
3. **✅ Automatic Discovery**: No custom code needed
4. **✅ Better Integration**: Works seamlessly with datasets library
5. **✅ Smaller Size**: Compressed Parquet files (~0.4MB total vs ~8MB original)

## 🔍 **Verification**

After upload, check that:
1. Dataset page loads: `https://huggingface.co/datasets/YOUR_USERNAME/fluke`
2. Dataset viewer works (you'll see a data preview)
3. Loading works: `load_dataset("YOUR_USERNAME/fluke")`

## 📋 **File Structure Uploaded**

```
fluke_dataset_standard/
├── README.md              # Dataset card with metadata
├── train.parquet          # Combined dataset (6,386 examples)
├── coref.parquet         # Coreference task (1,551 examples)
├── ner.parquet           # NER task (1,549 examples)
├── sa.parquet            # Sentiment task (1,644 examples)
├── dialogue.parquet      # Dialogue task (1,642 examples)
├── requirements.txt      # Dependencies
└── docs/
    └── USAGE.md         # Detailed usage guide
```

## 🆘 **Troubleshooting**

If you still encounter issues:

1. **Check file sizes**: Each Parquet file should be small (~100KB each)
2. **Verify format**: Use `pandas.read_parquet()` to test locally
3. **Test loading**: Try `load_dataset()` locally before uploading
4. **Repository settings**: Ensure repository is public
5. **Token permissions**: Verify your token has write access

## 🎉 **Success!**

Once uploaded, your FLUKE dataset will be:
- ✅ Discoverable on Hugging Face Hub
- ✅ Viewable in the browser
- ✅ Easy to load and use
- ✅ Professional and research-ready 