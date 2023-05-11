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

For ACE2005, we adopt 30 classes in ACE2005 which have more than 10 instances and randomly divide them into subsets with 10, 10 and 10 classes for training, validation and test, respectively.

For MAVEN , we adopt 100 classes in MAVEN which have more than 200 instances and randomly divide them into subsets with 64, 16 and 20 classes for training, validation and test, respectively.

For FewEvent, we adopt the version split by [Cong et al., 2021](https://aclanthology.org/2021.findings-acl.3.pdf), which contains 80, 10 and 10 event types for training, validation and test, respectively. 

| Dataset    | #Class | #Train | #Dev | #Test |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| ACE2005      | 30   | 10      | 10      | 10 |
| FewEvent  | 100      | 80     | 10 |20    |
| MAVEN   | 100      | 64       |  16| 10     |


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
