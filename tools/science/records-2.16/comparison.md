
Table 1: dimensions, and near and apart

| Vectors | Precision | Components for 95 % of the variance (features scaled) | Spectrum entropy, bits (scaled) | Random pairs, median cosine | One character changed | One value changed | Different benign requests | Text distance against vector distance, Spearman | Float against 8-bit vector, median cosine |
|---|---|---|---|---|---|---|---|---|---|
| published encoder | 8-bit, NPU | 2 (3) | 0.75 (1.22) | 0.83 | 0.9998 | 0.996 | 0.87 | 0.19 | – |
| random unit vectors | float | 121 (121) | – (6.99) | 0.00 | – | – | – | – | – |
| fixed stride | float | 25 (26) | 4.53 (4.61) | 0.03 | 0.9994 | 0.931 | 0.21 | 0.16 | – |
| fixed stride | 8-bit | 26 (26) | 4.57 (4.64) | 0.04 | 0.9969 | 0.919 | 0.25 | 0.16 | 0.994 |
| fixed stride, seed 1 | 8-bit | 29 (29) | 4.66 (4.75) | 0.02 | 0.9940 | 0.930 | -0.06 | 0.21 | 0.988 |
| delimiter | float | 16 (17) | 3.98 (4.07) | 0.11 | 0.9940 | 0.972 | 0.05 | 0.25 | – |
| delimiter | 8-bit | 17 (17) | 4.02 (4.11) | 0.11 | 0.9887 | 0.961 | 0.04 | 0.25 | 0.996 |
| delimiter, seed 1 | 8-bit | 16 (16) | 3.98 (4.07) | 0.10 | 0.9739 | 0.987 | -0.08 | 0.22 | 0.995 |
| entropy | float | 31 (32) | 4.72 (4.84) | 0.05 | 0.9571 | 0.681 | -0.17 | 0.07 | – |
| entropy | 8-bit | 31 (32) | 4.53 (4.69) | 0.06 | 0.9669 | 0.675 | -0.02 | 0.24 | 0.238 |
| entropy, seed 1 | 8-bit | 32 (33) | 4.74 (4.85) | 0.08 | 0.9818 | 0.580 | 0.00 | 0.17 | 0.278 |
| flat control | float | 23 (23) | 4.49 (4.55) | 0.06 | 0.9996 | 0.662 | 0.01 | 0.30 | – |
| flat control | 8-bit | 30 (31) | 4.72 (4.79) | 0.07 | 0.9958 | 0.488 | 0.02 | 0.25 | 0.950 |
| entropy + label | float | 32 (33) | 4.78 (4.88) | 0.07 | 0.9635 | 0.679 | -0.09 | 0.08 | – |
| entropy + label | 8-bit | 33 (34) | 4.74 (4.85) | 0.10 | 0.9808 | 0.525 | 0.03 | 0.17 | 0.278 |
| fixed stride, boundary drop | float | 13 (14) | 3.62 (3.72) | 0.11 | 0.9999 | 0.846 | -0.20 | 0.28 | – |
| fixed stride, boundary drop | 8-bit | 14 (14) | 3.71 (3.80) | 0.21 | 0.9987 | 0.861 | -0.03 | 0.32 | 0.966 |
| delimiter, boundary drop | float | 11 (11) | 3.38 (3.46) | 0.18 | 0.9914 | 0.981 | 0.06 | 0.35 | – |
| delimiter, boundary drop | 8-bit | 12 (13) | 3.50 (3.60) | 0.31 | 0.9803 | 0.974 | 0.08 | 0.34 | 0.907 |
| entropy, boundary drop | float | 19 (20) | 4.06 (4.17) | 0.08 | 0.9959 | 0.862 | -0.11 | 0.33 | – |
| entropy, boundary drop | 8-bit | 21 (22) | 4.20 (4.31) | 0.06 | 0.9741 | 0.880 | -0.06 | 0.34 | 0.960 |
| entropy, boundary drop | 8-bit, NPU | 22 (23) | 4.24 (4.34) | 0.07 | 0.9460 | 0.846 | -0.09 | 0.34 | – |

Table 2: structure, the held-out endpoints, feature ablation

| Vectors | Precision | Label shared with the 11 nearest | Endpoint shared with the 11 nearest (good texts) | Held-out linear head, ROC AUC | Held-out 11 neighbours, ROC AUC | Ablation, all 128 components: label ROC AUC / endpoint accuracy | First component only | Without the first 5 | Without the first 20 |
|---|---|---|---|---|---|---|---|---|---|
| published encoder | 8-bit, NPU | 0.922 | 0.798 | 0.939 | 0.791 | 0.939 / 0.802 | 0.936 / 0.756 | 0.517 / 0.798 | 0.442 / 0.634 |
| random unit vectors | float | 0.521 | 0.202 | 0.511 | 0.507 | 0.511 / 0.203 | – / – | – / – | – / – |
| fixed stride | float | 0.757 | 0.825 | 0.760 | 0.611 | 0.699 / 1.000 | 0.379 / 0.522 | 0.659 / 1.000 | 0.507 / 0.985 |
| fixed stride | 8-bit | 0.754 | 0.826 | 0.740 | 0.680 | 0.743 / 0.999 | 0.351 / 0.497 | 0.626 / 0.996 | 0.402 / 0.955 |
| fixed stride, seed 1 | 8-bit | 0.746 | 0.860 | 0.566 | 0.495 | 0.582 / 1.000 | 0.383 / 0.412 | 0.416 / 0.995 | 0.426 / 0.983 |
| delimiter | float | 0.702 | 0.793 | 0.769 | 0.660 | 0.644 / 0.959 | 0.711 / 0.347 | 0.515 / 0.954 | 0.245 / 0.784 |
| delimiter | 8-bit | 0.703 | 0.793 | 0.772 | 0.682 | 0.629 / 0.916 | 0.713 / 0.346 | 0.610 / 0.894 | 0.223 / 0.616 |
| delimiter, seed 1 | 8-bit | 0.697 | 0.788 | 0.795 | 0.518 | 0.745 / 0.871 | 0.195 / 0.400 | 0.673 / 0.863 | 0.303 / 0.639 |
| entropy | float | 0.727 | 0.966 | 0.743 | 0.717 | 0.670 / 1.000 | 0.302 / 0.645 | 0.591 / 0.997 | 0.621 / 0.966 |
| entropy | 8-bit | 0.705 | 0.967 | 0.641 | 0.686 | 0.618 / 1.000 | 0.477 / 0.703 | 0.598 / 0.962 | 0.489 / 0.782 |
| entropy, seed 1 | 8-bit | 0.699 | 0.955 | 0.631 | 0.621 | 0.659 / 0.999 | 0.408 / 0.467 | 0.589 / 0.949 | 0.559 / 0.850 |
| flat control | float | 0.734 | 0.936 | 0.705 | 0.547 | 0.598 / 1.000 | 0.612 / 0.636 | 0.625 / 0.991 | 0.409 / 0.916 |
| flat control | 8-bit | 0.728 | 0.933 | 0.672 | 0.495 | 0.620 / 1.000 | 0.605 / 0.616 | 0.608 / 0.924 | 0.429 / 0.800 |
| entropy + label | float | 0.736 | 0.976 | 0.782 | 0.748 | 0.828 / 1.000 | 0.251 / 0.661 | 0.791 / 0.997 | 0.676 / 0.961 |
| entropy + label | 8-bit | 0.713 | 0.976 | 0.714 | 0.714 | 0.722 / 1.000 | 0.289 / 0.533 | 0.615 / 0.966 | 0.397 / 0.830 |
| fixed stride, boundary drop | float | 0.767 | 0.811 | 0.760 | 0.702 | 0.724 / 1.000 | 0.330 / 0.401 | 0.624 / 0.998 | 0.451 / 0.960 |
| fixed stride, boundary drop | 8-bit | 0.763 | 0.813 | 0.746 | 0.698 | 0.760 / 0.992 | 0.332 / 0.459 | 0.748 / 0.976 | 0.584 / 0.747 |
| delimiter, boundary drop | float | 0.709 | 0.828 | 0.795 | 0.499 | 0.793 / 1.000 | 0.711 / 0.454 | 0.517 / 0.996 | 0.262 / 0.968 |
| delimiter, boundary drop | 8-bit | 0.705 | 0.836 | 0.783 | 0.497 | 0.780 / 0.981 | 0.807 / 0.448 | 0.730 / 0.931 | 0.293 / 0.639 |
| entropy, boundary drop | float | 0.736 | 0.985 | 0.741 | 0.585 | 0.744 / 1.000 | 0.407 / 0.709 | 0.713 / 0.991 | 0.469 / 0.886 |
| entropy, boundary drop | 8-bit | 0.718 | 0.978 | 0.781 | 0.572 | 0.803 / 1.000 | 0.408 / 0.673 | 0.799 / 0.945 | 0.586 / 0.645 |
| entropy, boundary drop | 8-bit, NPU | 0.722 | 0.977 | 0.737 | 0.564 | 0.715 / 0.999 | 0.471 / 0.617 | 0.693 / 0.950 | 0.513 / 0.656 |

Table 3: folding, and the triage quality of the science gate

| Vectors | Precision | Distinct vectors of the 12,078 texts of the analysis set | Distinct vectors of all distinct inputs | Vectors shared by different inputs (inputs that share) | Split of the 1.7 note, all texts: head / neighbours | Five folds by path, pooled: head / neighbours |
|---|---|---|---|---|---|---|
| published encoder | 8-bit, NPU | 11,051 | 11,052 of 12,079 | 381 (1,408) | 0.940 / 0.551 | – / – |
| random unit vectors | float | – | – of – | – (–) | 0.500 / 0.507 | 0.494 / 0.506 |
| fixed stride | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.692 / 0.569 | 0.613 / 0.587 |
| fixed stride | 8-bit | 11,947 | 25,362 of 25,608 | 182 (428) | 0.732 / 0.662 | 0.617 / 0.598 |
| fixed stride, seed 1 | 8-bit | 11,898 | 25,249 of 25,608 | 263 (622) | 0.461 / 0.492 | 0.547 / 0.602 |
| delimiter | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.641 / 0.664 | 0.583 / 0.643 |
| delimiter | 8-bit | 12,075 | 25,598 of 25,608 | 10 (20) | 0.637 / 0.575 | 0.580 / 0.632 |
| delimiter, seed 1 | 8-bit | 12,072 | 25,587 of 25,608 | 21 (42) | 0.614 / 0.518 | 0.603 / 0.639 |
| entropy | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.660 / 0.652 | 0.584 / 0.622 |
| entropy | 8-bit | 11,760 | 25,042 of 25,608 | 389 (955) | 0.584 / 0.637 | 0.567 / 0.554 |
| entropy, seed 1 | 8-bit | 11,696 | 24,879 of 25,608 | 491 (1,220) | 0.517 / 0.604 | 0.560 / 0.547 |
| flat control | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.743 / 0.595 | 0.667 / 0.632 |
| flat control | 8-bit | 10,505 | 22,452 of 25,608 | 1,218 (4,374) | 0.693 / 0.598 | 0.665 / 0.638 |
| entropy + label | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.728 / 0.734 | 0.692 / 0.688 |
| entropy + label | 8-bit | 11,861 | 25,193 of 25,608 | 324 (739) | 0.629 / 0.614 | 0.615 / 0.603 |
| fixed stride, boundary drop | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.729 / 0.683 | 0.578 / 0.653 |
| fixed stride, boundary drop | 8-bit | 11,915 | 25,254 of 25,608 | 287 (641) | 0.745 / 0.676 | 0.579 / 0.661 |
| delimiter, boundary drop | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.752 / 0.495 | 0.681 / 0.637 |
| delimiter, boundary drop | 8-bit | 12,019 | 25,462 of 25,608 | 135 (281) | 0.743 / 0.506 | 0.675 / 0.620 |
| entropy, boundary drop | float | 12,078 | 25,608 of 25,608 | 0 (0) | 0.660 / 0.599 | 0.559 / 0.638 |
| entropy, boundary drop | 8-bit | 11,658 | 24,712 of 25,608 | 517 (1,413) | 0.756 / 0.555 | 0.540 / 0.615 |
| entropy, boundary drop | 8-bit, NPU | 11,717 | 24,827 of 25,608 | 488 (1,269) | – / – | – / – |
