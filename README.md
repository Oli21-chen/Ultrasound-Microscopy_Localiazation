# Bubble_Localization_SR
individual project-updating
1. original network structure  -encoderdecoder
2. Different architecture  - skip connection/ residual connection
3. 0% overlap of patches  - cutting image/ concatenation


In final version:
1. X_train needs to normalize before training, Y_train does not
2. Binaraize label image - depends
3. Different training data require fine-tuning parameters

2022May version: This is updated for ICL. 
1. Updated the HRNETV2
2. Updated the JNet without Concatenation. 
