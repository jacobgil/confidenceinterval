# The long missing python library for confidence intervals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Build Status](https://github.com/jacobgil/confidenceinterval/workflows/Tests/badge.svg)
[![Downloads](https://static.pepy.tech/personalized-badge/confidenceinterval?period=month&units=international_system&left_color=black&right_color=brightgreen&left_text=Monthly%20Downloads)](https://pepy.tech/project/confidenceinterval)

`pip install confidenceinterval`

This is a package that computes common machine learning metrics like F1, and returns their confidence intervals.

⭐ Very easy to use, with the standard scikit-learn naming convention and interface:

e.g roc_auc_score(y_true, y_pred).

⭐ Support for many metrics, with modern confidence interval methods.

⭐ Support for both analytical computation of the confidence intervals, and bootstrapping.

⭐ East to use interface to compute confidence intervals on new metrics that don't appear here, with bootstrapping.

## Getting started

```python
from confidenceinterval import roc_auc_score
auc, ci = roc_auc_score(y_true, y_pred, confidence_level=0.95)
auc, ci = roc_auc_score(y_true, y_pred, confidence_level=0.95, method='bootstrap_bca')
```

By default all the methods return an analytical computation of the confidence interval (CI).
For a bootstrap computation of the CI, for any of the methods belonw, you can just specify method='bootstrap_bca', or method='bootstrap_percentile' or method='bootstrap_basic'.

## Supported methods

### Get a confidence interval for any external metric
With the bootstrap_ci method, you can get the CI for a an external metric method.
As an example, lets get the CI for the balanced accuracy metric. It's not implemented yet in this package,
but we can easily get the CI for it:

```python
from confidenceinterval.bootstrap import bootstrap_ci
# You can specify a random generator for reproducability, or pass None
random_generator = np.random.default_rng()
bootstrap_ci(y_true=y_true,
             y_pred=y_pred,
             metric=sklearn.metrics.balanced_accuracy_score,
             confidence_level=0.95,
             n_resamples=9999,
             method='bootstrap_bca',
             random_state=random_generator)
```

### F1, Precision, Recall (with Macro and Micro averaging)
```python
from confidence interval import precision_score, recall_score, f1_score
```

These methods also accept average='micro' or average='macro'.

The analytical computation here is using the (amazing) 2022 paper of Takahashi et al (reference below). 


### ROC AUC
```python
from confidence interval import roc_auc_score
```
The analytical computation here is a fast implementation of the DeLong method.


### Binary metrics
```python
from confidence interval import accuracy_score, ppv_score, npv_score,
                                tpr_score, fpr_score, tnr_score
```

For these methods, the confidence interval is estimated by treating the ratio as a binomial proportion,
see the [wiki page](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval).

By default method='wilson', the wilson interval, which behaves better for smaller samples.

method can be one of ['wilson', 'normal', 'agresti_coull', 'beta', 'jeffreys', 'binom_test'], or one of the boostrap methods.

----------

### References

The binomial confidence interval computation uses the statsmodels package:
https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html

Yandex data schhol implementation of the fast delong method:
https://github.com/yandexdataschool/roc_comparison

https://ieeexplore.ieee.org/document/6851192
X. Sun and W. Xu, "Fast Implementation of DeLong’s Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves," in IEEE Signal Processing Letters, vol. 21, no. 11, pp. 1389-1393, Nov. 2014, doi: 10.1109/LSP.2014.2337313.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8936911/#APP2

Confidence interval for micro-averaged F1 and macro-averaged F1 scores
`Kanae Takahashi,1,2 Kouji Yamamoto,3 Aya Kuchiba,4,5 and Tatsuki Koyama6`

B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall/CRC, Boca Raton, FL, USA (1993)

http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf

`Nathaniel E. Helwig, “Bootstrap Confidence Intervals”`

Bootstrapping (statistics), Wikipedia, https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29