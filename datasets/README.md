# Downloaded Datasets

This directory contains datasets for the humor recognition research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset Overview

| Dataset | Size | Task | Format |
|---------|------|------|--------|
| ColBERT Humor | 200k | Binary classification | HuggingFace Dataset |
| Short Jokes | 231k | Joke corpus | HuggingFace Dataset |
| Dad Jokes | 53k | Q&A jokes | HuggingFace Dataset |
| Reddit Jokes | 100k | Jokes with scores | HuggingFace Dataset |

---

## Dataset 1: ColBERT Humor Detection (PRIMARY)

### Overview
- **Source**: HuggingFace: `CreativeLang/ColBERT_Humor_Detection`
- **Size**: 200,000 samples (100k humor, 100k non-humor)
- **Format**: HuggingFace Dataset
- **Task**: Binary humor classification
- **Splits**: train only (need to create own splits)
- **License**: Not specified

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("CreativeLang/ColBERT_Humor_Detection")
dataset.save_to_disk("datasets/colbert_humor")
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/colbert_humor")
print(dataset['train'][0])
# {'text': '...', 'humor': True/False}
```

### Sample Data

```json
{
  "positive_samples": [
    {"text": "What do you call a turtle without its shell? dead.", "humor": true},
    {"text": "My life is like a romantic comedy except there's no romance and it's just me laughing at my own jokes.", "humor": true}
  ],
  "negative_samples": [
    {"text": "Joe biden rules out 2020 bid: 'guys, i'm not running'", "humor": false},
    {"text": "Watch: darvish gave hitter whiplash with slow pitch", "humor": false}
  ]
}
```

### Notes
- Balanced dataset (50/50 humor/non-humor)
- Short texts (10-200 characters)
- Non-humor samples are news headlines
- Primary dataset for experiments

---

## Dataset 2: Short Jokes

### Overview
- **Source**: HuggingFace: `ysharma/short_jokes`
- **Size**: 231,657 jokes
- **Format**: HuggingFace Dataset (CSV backend)
- **Task**: Joke generation/positive class for classification
- **Splits**: train only
- **License**: Not specified

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("ysharma/short_jokes")
dataset.save_to_disk("datasets/short_jokes")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/short_jokes")
print(dataset['train'][0])
# {'ID': 1, 'Joke': '...'}
```

### Sample Data

```json
[
  {"ID": 1, "Joke": "[me narrating a documentary about narrators] \"I can't hear what they're saying cuz I'm talking\""},
  {"ID": 2, "Joke": "Telling my daughter garlic is good for you. Good immune system and keeps pests away. Ticks, mosquitos, vampires... men."}
]
```

### Notes
- All positive class (jokes only)
- Would need to pair with non-joke text for classification
- Good for joke generation tasks

---

## Dataset 3: Dad Jokes

### Overview
- **Source**: HuggingFace: `shuttie/dadjokes`
- **Size**: 53,400 total (52,000 train, 1,400 test)
- **Format**: HuggingFace Dataset
- **Task**: Joke completion (question → response)
- **Splits**: train, test
- **License**: Not specified

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("shuttie/dadjokes")
dataset.save_to_disk("datasets/dadjokes")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/dadjokes")
print(dataset['train'][0])
# {'question': '...', 'response': '...'}
```

### Sample Data

```json
[
  {"question": "I asked my priest how he gets holy water", "response": "He said it's just regular water, he just boils the hell out of it"},
  {"question": "Life Hack: If you play My Chemical Romance loud enough in your yard", "response": "your grass will cut itself"}
]
```

### Notes
- Setup-punchline format
- Good for studying joke structure
- Smaller but cleaner dataset

---

## Dataset 4: Reddit Jokes (Subset)

### Overview
- **Source**: HuggingFace: `SocialGrep/one-million-reddit-jokes`
- **Size**: 100,000 (subset of 1M)
- **Format**: HuggingFace Dataset
- **Task**: Jokes with popularity scores
- **Splits**: train only
- **License**: Not specified

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("SocialGrep/one-million-reddit-jokes")
# Save subset to save space
subset = dataset['train'].select(range(100000))
subset.save_to_disk("datasets/reddit_jokes_100k")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/reddit_jokes_100k")
print(dataset[0])
# {'title': '...', 'selftext': '...', 'score': int, ...}
```

### Sample Data

```json
[
  {"title": "Why did the programmer quit his job?", "selftext": "Because he didn't get arrays.", "score": 150},
  {"title": "I told my wife she was drawing her eyebrows too high", "selftext": "She looked surprised.", "score": 2500}
]
```

### Notes
- Has upvote scores (humor quality proxy)
- Variable text length
- May contain some low-quality/offensive content

---

## Sample Data File

A `samples.json` file is included with example records from each dataset for quick reference without loading full datasets.

---

## Data Splits for Experiments

The ColBERT dataset comes as a single train split. For experiments, create splits:

```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/colbert_humor")['train']

# Shuffle and split
dataset = dataset.shuffle(seed=42)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))

train = dataset.select(range(train_size))
val = dataset.select(range(train_size, train_size + val_size))
test = dataset.select(range(train_size + val_size, len(dataset)))

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
# Train: 160000, Val: 20000, Test: 20000
```

---

## Storage Notes

- Total storage: ~500MB for all datasets
- ColBERT: ~50MB
- Short Jokes: ~60MB
- Dad Jokes: ~15MB
- Reddit Jokes (100k): ~100MB

Data files are excluded from git via `.gitignore`. Re-download using instructions above if needed.
