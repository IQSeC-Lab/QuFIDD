Validation of the PyTorch port against CADE's released code, model and results (Drebin, FakeDoc withheld = drebin_new_7).
Needs a clone of github.com/S-R-Nowmi/CADE (data/, models/, reports/) and h5py:
  git clone https://github.com/S-R-Nowmi/CADE ~/CADE && pip install h5py
  CADE_REPO=~/CADE python3 validate.py              # metrics on CADE's saved masks + our CADE mask port (~5 min)
  CADE_REPO=~/CADE python3 validate.py --skip-cade --given-g   # our boundary mask with CADE's saved approximation MLPs
  CADE_REPO=~/CADE python3 val_boundary.py          # our full boundary baseline, 4 seeds
  CADE_REPO=~/CADE python3 val_coin.py              # our COIN port (~4 min)
Results obtained (CADE paper Tables 4/6 in brackets):
  CADE encoder reload: centroids within 3e-5, MAD within 7e-5 of CADE's saved values; closest families identical
  original distance                         5.363 +- 0.568          [5.363 +- 0.568]
  CADE's saved masks, our metric code       k 44.693 +- 6.151, d 0.065 +- 0.035, cross 97.64%   [identical]
  CADE's saved boundary masks, our metric   d 3.960 +- 2.963, cross 0.00%                       [identical]
  random (100 x)                            d 5.384, cross 0.00%    [5.423 +- 1.773, 0.00%]
  our CADE mask optimiser                   k 43.8 +- 5.7, d 0.078 +- 0.049, cross 94.49%     [44.7, 0.065, 97.64%]
  our boundary mask with CADE's g           d 3.861 +- 3.041, cross 0.00%                       [3.960 +- 2.963, 0.00%]
  our full boundary baseline (4 seeds)      d 4.65 / 7.92 / 6.80 / 6.19, cross 0-0.8%           [3.960, 0.00%]
  our COIN port                             d 4.952 +- 1.689, cross 0.00%                       [6.219 +- 3.962, 0.00%]
The boundary baseline varies strongly with the random training of its approximation MLP (the mask step reproduces CADE
when given CADE's MLP); COIN is stochastic (synthetic sampling, KMeans) and CADE reports its best converged average.
