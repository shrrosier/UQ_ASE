This repository contains code related a paper in the process of being submitted, entitled 'Calibrated sea level contribution from the Amundsen Sea sector, West Antarctica, under RCP8.5 and Paris 2C scenarios'. 

The two directories relate to two separate surrogate models:
- RNN_surrogate: A recurrent neural network surrogate model that predicts change in ice surface speed and elevation for given input model parameters for the years 1998-2018.
- LSTM_surrogate: A Long-short term memory surrogate model that predicts change in ice volume above flotation for given input model parameters for the years 2021-2100.

Both networks are trained on output from the ice sheet model Úa, freely accessible here: https://github.com/GHilmarG/UaSource
