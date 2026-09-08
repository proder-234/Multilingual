
=== Overall vs. English ===
{'precision': 0.5683646112600537, 'recall': 0.7737226277372263, 'f1': 0.6553323029366306, 'confusion_matrix': [[165, 161], [62, 212]]}

=== Aggregate accuracy vs. English ===
0.6283333333333333

=== Per-language metrics vs. English ===
  language    n  accuracy  precision    recall        f1
1   Nepali  300  0.600000   0.544041  0.766423  0.636364
0    Hindi  300  0.656667   0.594444  0.781022  0.675079

=== Translation invariance ===
0.52

=== Language gap vs. English ===
  language    n        f1    f1_gap  accuracy  accuracy_gap
1   Nepali  300  0.636364  0.363636  0.600000      0.400000
0    Hindi  300  0.675079  0.324921  0.656667      0.343333

=== Pairwise language agreement ===
  language_1 language_2    n  agreement
1    English     Nepali  300   0.600000
0    English      Hindi  300   0.656667
2      Hindi     Nepali  300   0.783333

=== Per-language F1 variance ===
f1_mean    0.655721
f1_std     0.027376
f1_min     0.636364
f1_max     0.675079
Name: f1, dtype: float64

=== Fleiss' kappa across languages ===
0.34841628959276033