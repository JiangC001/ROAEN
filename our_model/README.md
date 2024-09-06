# ROAEN-ABSA
ROAEN: Reversed dependency graph and Orthogonal-gating strategy Attention-Enhanced Network for Aspect-Level Sentiment Classification  
2024

## Requirements

- torch==1.4.0
- scikit-learn==0.23.2
- transformers==3.2.0
- cython==0.29.13
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.

## Preparation

2. Prepare dataset with:

   `python preprocess_data.py`

3. Prepare vocabulary with:

   `sh build_vocab.sh`

## Training

To train the ROAEN model, run:

`sh run.sh`

## Credits

The code and datasets in this repository are based on [SSEGCN_ABSA-main](https://github.com/zhangzheng1997/SSEGCN-ABSA) .

