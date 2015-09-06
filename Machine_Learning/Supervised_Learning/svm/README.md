Use SVM to identify emails from the Enron corpus by their authors and to find accuracy and speed of different Kernels. <br />
<br />
**Linear Kernel with full dataset.** <br />
Accuracy: .98 <br />
Training Time: 230 seconds <br />
Testing Time: 32 seconds <br />

**Linear Kernel with 1/100 of dataset.** <br />
Accuracy: 0.884527872582 <br />
Training Time: .122 seconds <br />
Testing Time: 1.298 seconds <br />

**RBF Kernel with 1/100 of dataset.** <br />
Accuracy: .616040955631 <br />
Training Time: .117 seconds <br />
Testing Time: 1.254 seconds <br />

**RBF Kernel with 1/100 of dataset with different C (10,100,1000,10000). <br />
Finally best C value was 10000.** <br />
Accuracy: 0.892491467577 <br />
Training Time: 0.11 seconds <br />
Testing Time: 1.017 seconds <br />

**RBF Kernel with full dataset with C = 10000** <br />
Accuracy: .990898748578 <br />
Training Time: 121.139 seconds <br />
Testing Time: 15.2 seconds <br />

**Email outputs:** <br />
10th email prediction: 1 (Chris) <br />
26th email prediction: 0 (Sarah) <br />
50th email prediction: 1 (Chris) <br />

Predicted emails from Chris = 877 <br />
