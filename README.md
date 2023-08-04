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

For ERE, we adopt the top 30 classes with the most samples and split into 10/10/10 for training, validation and test, respectively.

| Dataset    | #Class | #Train | #Dev | #Test |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| ACE2005      | 30   | 10      | 10      | 10 |
| ERE     | 30   | 10      | 10      | 10 |
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


## Results (F1 scores)

We run each experiment 5 times with 5 five different seeds to get the averages and standard deviations for fair comparison.

### ACE2005

| **Model**      | **5-way-5-shot** | **5-way-10-shot** | **10-way-5-shot** | **10-way-10-shot** |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Match      | 24.29 $\pm$ 0.22 | 40.21 $\pm$ 0.41  | 19.29 $\pm$ 0.18  | 26.91 $\pm$ 0.08 |
| Proto      | 53.49 $\pm$ 0.41 | 34.05 $\pm$ 0.94  | **50.63** $\pm$ **0.10**   | 32.66 $\pm$ 0.17 |
| Proto-dot      | 23.59 $\pm$ 0.42 | 19.46 $\pm$ 0.11 | 14.09 $\pm$ 0.05  | 11.03 $\pm$ 0.09 |
| Relation      | 11.25 $\pm$ 0.21 | 23.33 $\pm$ 0.56  | 5.33 $\pm$ 0.22   | 12.55 $\pm$ 0.14 |
| VanillaCRF      | 27.55 $\pm$ 0.56 | 41.75 $\pm$ 0.46  | 11.35 $\pm$ 0.05  | 15.86 $\pm$ 0.11 |
| PACRF      | 52.61 $\pm$ 0.36 | 42.18 $\pm$ 1.52  | 36.78 $\pm$ 0.41  | 28.42 $\pm$ 0.32 |
| SpanFSED     | **56.22** $\pm$ **0.33** | **54.53** $\pm$ **0.22**  | 49.62 $\pm$ 0.26   | **50.76** $\pm$ **0.36** |

### ERE

| **Model**      | **5-way-5-shot** | **5-way-10-shot** | **10-way-5-shot** | **10-way-10-shot** |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Match       | 34.38 $\pm$ 0.28 | 39.21 $\pm$ 0.27  | 21.98 $\pm$ 0.09  | 33.03 $\pm$ 0.08 |
| Proto       | 45.33 $\pm$ 0.25 | 48.00 $\pm$ 0.31  | 45.23 $\pm$ 0.10  | 39.36 $\pm$ 0.09 |
| Proto-dot   | 31.48 $\pm$ 0.37 | 34.68 $\pm$ 0.25  | 21.99 $\pm$ 0.09  | 24.69 $\pm$ 0.07 |
| Relation    | 21.93 $\pm$ 0.37 | 23.79 $\pm$ 0.34  | 12.50 $\pm$ 0.15  | 19.14 $\pm$ 0.12 |
| VanillaCRF  | 30.91 $\pm$ 0.49 | 51.58 $\pm$ 0.28  | 27.34 $\pm$ 0.04  | 25.48 $\pm$ 0.10 |
| PACRF       | 49.57 $\pm$ 0.33 | 63.33 $\pm$ 0.21  | **46.67** $\pm$ **0.30**  | 39.68 $\pm$ 0.16 |
| SpanFSED    | **51.78** $\pm$ **0.22** | **67.09** $\pm$ **0.27**  | 45.21 $\pm$ 0.19   | **47.48** $\pm$ **0.15** |

### FewEvent

| **Model**      | **5-way-5-shot** | **5-way-10-shot** | **10-way-5-shot** | **10-way-10-shot** |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Match    | 28.46 $\pm$ 1.01 | 37.81 $\pm$ 2.32  | 26.58 $\pm$ 0.52   | 29.25 $\pm$ 0.40|
| Proto      | 36.38 $\pm$ 1.50 | 56.27 $\pm$ 0.70  | 17.76 $\pm$ 1.81   | 43.21 $\pm$ 1.02 |
| Proto-dot    |  40.39 $\pm$ 0.37 | 53.44 $\pm$ 1.33 | 46.65 $\pm$ 0.68  | 56.24 $\pm$ 0.26 |
| Relation      | 10.68 $\pm$ 0.49 | 28.66 $\pm$ 0.49  | 9.55 $\pm$ 0.48  | 14.14 $\pm$ 0.05 |
| VanillaCRF    | 44.12 $\pm$ 1.02 | 61.13 $\pm$ 1.10  | 41.61 $\pm$ 0.21 | 51.94 $\pm$ 0.29 |
| PACRF      | 47.76 $\pm$ 1.11 | 67.38 $\pm$ 0.80 | 43.22 $\pm$ 0.70  | 54.26 $\pm$ 1.30 | 
| SpanFSED    | **63.63** $\pm$ **0.38** |**72.59** $\pm$ **0.33**  | **65.34** $\pm$ **0.26**  | **75.42** $\pm$ **0.20** |

### MAVEN

| **Model**      | **5-way-5-shot** | **5-way-10-shot** | **10-way-5-shot** | **10-way-10-shot** |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| Match    | 47.37 $\pm$ 0.33 | 59.01 $\pm$ 0.05  | 24.54 $\pm$ 0.25   | 49.72 $\pm$ 0.31 |
| Proto      | 72.10 $\pm$ 0.19 | 77.17 $\pm$ 0.12 | 67.43 $\pm$ 0.08   | 60.75 $\pm$ 0.12 |
| Proto-dot    | 69.76 $\pm$ 0.66 | 72.25 $\pm$ 0.76 | 56.06 $\pm$ 1.00  | 58.56 $\pm$ 1.21 |
| Relation      | 38.10 $\pm$ 0.43 | 39.36 $\pm$ 0.73  | 39.69 $\pm$ 0.18   | 42.65 $\pm$ 0.18 |
| VanillaCRF   | 63.05 $\pm$ 0.83 | 69.41 $\pm$ 0.04| 58.89 $\pm$ 0.66 | 64.32 $\pm$ 0.39 |
| PACRF      | 77.06 $\pm$ 0.23 | 75.29 $\pm$ 0.65 | 73.89 $\pm$ 0.12  | 75.73 $\pm$ 0.12 |
| SpanFSED    | **77.91** $\pm$ **0.16** | **78.89** $\pm$ **0.66**  | **74.64** $\pm$ **0.12**   | **76.14** $\pm$ **0.17** |
