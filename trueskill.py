#!/usr/bin/env python2.6
from __future__ import print_function
import math
import sys

from scipy.stats.distributions import norm as scipy_norm

beta = 25/6
gamma = 25/300
epsilon = 0.08

norm = scipy_norm()

def pdf(x):
    return norm.pdf(x)

def cdf(x):
    return norm.cdf(x)

def Vwin(t, e):
    return pdf(t - e) / cdf(t - e)

def Wwin(t, e):
    return Vwin(t, e) * (Vwin(t, e) + t - e)

def true_skill(winner, loser):
    muw, sigmaw = winner
    mul, sigmal = loser
    
    c = (2*beta**2 + sigmaw**2 + sigmal**2)**.5
    t = (muw - mul) / c
    e = epsilon / c
    
    sigmaw_new = (sigmaw**2 * (1 - (sigmaw**2) / (c**2)*Wwin(t, e)) + gamma**2)**.5
    sigmal_new = (sigmal**2 * (1 - (sigmal**2) / (c**2)*Wwin(t, e)) + gamma**2)**.5
    muw_new = (muw + sigmaw**2/c * Vwin(t,e))
    mul_new = (mul - sigmal**2/c * Vwin(t,e))
    
    winner = muw_new, sigmaw_new
    loser = mul_new, sigmal_new
    
    return winner, loser
  
def update(squads, winner, loser):
    winner_stats = squads.get(winner, (25, 25/3))
    loser_stats = squads.get(loser, (25, 25/3))
    squads[winner], squads[loser] = true_skill(winner_stats, loser_stats)

def rank(squads, squad):
    return squads[squad][0] - 3*squads[squad][1]

def main(argv):
    games = [
        ("PK", "SF")   , ("RT", "XT")  , ("ACE", "BS") , ("SRM", "GB") , #day 1
        ("SysX", "XT") , ("PK", "RT")  , ("SF", "GB")  , ("SRM", "BS") , #day 2
        ("SysX", "GB") , ("RT", "BS")  , ("SF", "SRM") , ("XT", "ACE") , #day 3
        ("SysX", "PK") , ("RT", "SF")  , ("XT", "BS")  , ("GB", "SRM") , #day 4
        ("SysX", "RT") , ("PK", "ACE") , ("XT", "SF")  , ("BS", "GB")  , #day 5
        ("SysX", "SF") , ("RT", "GB")  , ("XT", "PK")  , ("ACE", "SRM"), #day 6
        ("SysX", "ACE"), ("RT", "SF")  , ("XT", "PK")  , ("BS", "SRM") , #day 7
        ("SysX", "BS") , ("PK", "GB")  , ("RT", "SRM") , ("ACE", "SF") , #day 8
        ("SysX", "RT") , ("PK", "SRM") , ("XT", "SF")  , ("BS", "ACE") , #day 9
        ("PK", "SF")   , ("XT", "SysX"), ("SRM", "ACE"), ("BS", "GB")  , #day 10
        ("SysX", "SF") , ("PK", "RT")  , ("XT", "SRM") , ("ACE", "GB") , #day 11
        ("SysX", "SRM"), ("PK", "BS")  , ("RT", "ACE") , ("XT", "GB")  , #day 12
        ("SysX", "PK") , ("RT", "XT")  , ("BS", "ACE") , ("GB", "ACE") ] #day 13

    squads = {}
    
    for winner, loser in games:
        update(squads, winner, loser)
    
    for squad in sorted(squads, key=lambda squad: rank(squads, squad)):
        mu = squads[squad][0]
        sigma = squads[squad][1]
        print("%6s R: %5.2f mu: %5.2f sigma: %5.2f" % (squad, rank(squads, squad), mu, sigma))
    
    return 0

if __name__ == "__main__":
    exit_code = main(sys.argv)
    sys.exit(exit_code)
