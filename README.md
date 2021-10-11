# spectracl
Model based classification of MALDI-TOF spectra.

# Method
* standarise and make spectra comparable by:
    * square root transformation
    * smoothing
    * baseline removal
    * normalisation/calibration
    * taking mean of bins with width 1
    * selecting a pre-determined set of features
* perform classification with model
