# COMP 657 TME 2
Jamie Czerwinski

This repo represents my admittedly very basic attempt to get [river](https://riverml.xyz/latest/) working. I was able to start running and playing with a [matrix factorization model for recommender systems as tutorialed on the river docs](https://riverml.xyz/latest/examples/matrix-factorization-for-recommender-systems-part-1/).

I basically ran all the steps in Part 1 of the tutorial without any changes.

```sh
git clone https://github.com/jczerwinski/river-play.git
cd river-play
pipenv install git+https://github.com/online-ml/river#egg=river sklearn
pipenv shell
py test.py
```
