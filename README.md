#eep-class-aware-ood explores uncertainty estimation for open-set recognition on the Oxford-102 Flowers dataset. The main contribution is a simple but surprisingly effective cross-class validation procedure for discovering OOD thresholds using only in-distribution labels.

Core Steps:

Hold out one class from the set of “known” classes.

Compute uncertainty scores for:

ID data (remaining classes)

OOD proxy data (held-out class)

Select the uncertainty threshold that maximizes F1 for separating ID vs the proxy OOD class.

Repeat over all classes and compute the best threshold averaged over splits.

This procedure works regardless of the chosen uncertainty metric—entropy, softmax energy, distance to class centroids, feature-space density estimates, etc.
![Overview of our approach](approach.png)
