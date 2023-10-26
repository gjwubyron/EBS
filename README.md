# Zero-shot Fake News Detection

This is the repository for the project in Essentials in Text and Speech Processing.

## Setup

We recommend start with a clean environment:

```bash
conda create -n fnd python=3.9
conda activate fnd
```

Install required packages:

```bash
git clone https://github.com/gjwubyron/EBS.git
cd EBS
pip install -r requirements.txt
```

## Usage

### Data

We use two datasets from MultiFC. You can download the datasets from <https://github.com/casperhansen/fake-news-reasoning>.

### Run

```bash
python main.py --dataset {pomt, snes}

```

## View results

We also provide the [jupyter notebook](https://github.com/gjwubyron/EBS/blob/master/view.ipynb) to view the results.
