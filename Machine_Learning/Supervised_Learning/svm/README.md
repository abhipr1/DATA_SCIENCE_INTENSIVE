Use SVM to identify emails from the Enron corpus by their authors and to find accuracy and speed of different Kernels.

==========================================================
Linear Kernel with full dataset.
Accuracy: .98
Training Time: 230 seconds
Testing Time: 32 seconds
==========================================================
Linear Kernel with 1/100 of dataset.
Accuracy: 0.884527872582
Training Time: .122 seconds
Testing Time: 1.298 seconds
==========================================================
RBF Kernel with 1/100 of dataset.
Accuracy: .616040955631
Training Time: .117 seconds
Testing Time: 1.254 seconds
==========================================================
RBF Kernel with 1/100 of dataset with different C (10,100,1000,10000).
Finally best C value was 10000.
Accuracy: 0.892491467577
Training Time: 0.11 seconds
Testing Time: 1.017 seconds
==========================================================
RBF Kernel with full dataset with C = 10000
Accuracy: .990898748578
Training Time: 121.139 seconds
Testing Time: 15.2 seconds
==========================================================
Email outputs:
10th email prediction: 1 (Chris)
26th email prediction: 0 (Sarah)
50th email prediction: 1 (Chris)

Predicted emails from Chris = 877
