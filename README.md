# Few Shot Event Detection

## Environment

- matplotlib==3.6.2
- numpy==1.23.5
- scikit_learn==1.2.2
- torch==1.10.0
- tqdm==4.64.1
- transformers==4.28.1

```python
pip install -r requirements.txt
```

## Dataset


| Dataset    | #Class | #Train | #Dev | #Test |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| ACE2005      | 33   | 13      | 10      | 10 |
| FewEvent  | 100      | 80     | 10 |20    |
| Maven   | 100      | 64       |  16| 10     |


## Models

Metric-based:

- **Match**: uses cosine function to measure the similarity.

- **Proto**: uses Euclidean Distance as the similarity metric.

- **Proto-dot**: uses dot product to compute the similarity

- **Relation**: building a two-layer neural networks to measure the similarity.

CRF-based:

- **Vanilla CRF**: adopt the vanilla CRF in the FSED task without considering the adaptation problem.
- **PA-CRF**: exploring event type dependencies via Gaussian distribution for approximation.


## Results

| **Model**      | **5-way-5-shot** | **5-way-10-shot** | **10-way-5-shot** | **10-way-10-shot** |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Match      | (-,-,-) | (-,-,-)  | (-,-,-)   | (-,-,-) |
| Proto      | (-,-,-) | (-,-,-)  | (-,-,-)   | (-,-,-) |
| Proto-dot      | (-,-,-) | (-,-,-)  | (-,-,-)   | (-,-,-) |
| Relation      | (-,-,-) | (-,-,-)  | (-,-,-)   | (-,-,-) |
| VanillaCRF      | (-,-,-) | (-,-,-)  | (-,-,-)   | (-,-,-) |
| PACRF      | (-,-,-) | (-,-,-)  | (-,-,-)   | (-,-,-) |
| ETS     | (-,-,-) | (-,-,-)  | (-,-,-)   | (-,-,-) |