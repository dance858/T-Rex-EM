# T-Rex-EM
This repository contains code for **T-Rex**, an expectation-maximization (EM) algorithm for robustly fitting a statistical factor model. It fits a factor model by solving 
the maximum likelihood estimation problem

$$
\begin{array}{r}
\text{minimize} & \text{log det } (FF^T + D) + (n/m) \sum_{i=1}^m \log(x_i^T (FF^T + D)^{-1} x_i) \\
\end{array}
$$ 

where the variables are the diagonal matrix $D \in \mathbf{R}^{n \times n}$ and $F \in \mathbf{R}^{n \times r}.$ For more information, see our [paper]().

## Recreating the figures in the paper
First install the packages listed in 'requirements.txt'.

* To recreate Figure 1, run `MSE_vs_samples.ipynb`.
* To recreate Figure 2, run `MSE_vs_samples.ipynb`.
* To recreate Figure 3 and 4, run `TRex_vs_Tyler.ipynb`. 
* To recreate Figure 5 and 6, run `collect_face_data.py` followed by `plot_recovered_faces.py` and `plot_all_test_faces.py`.

## Citing
If you wish to cite this work you may use the following BibTex:

```
@article{Cederberg25,
title = {},
journal = {},
volume = {},
pages = {},
year = {},
issn = {},
doi = {},
author = {},
}
```


