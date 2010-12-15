#!/usr/bin/env python2.6
from __future__ import print_function
import math
import sys
import csv

from scipy.stats.distributions import norm as scipy_norm
import numpy as np

beta = 21.
gamma = 1./12
epsilon = 0.01
STDEVS = 2
MEAN = 25.
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

def update(player_ranks, winner, loser):
    winner_stats = player_ranks.get(winner, (MEAN, MEAN/STDEVS))
    loser_stats = player_ranks.get(loser, (MEAN, MEAN/STDEVS))
    player_ranks[winner], player_ranks[loser] = true_skill(winner_stats, loser_stats)

def rank(player_ranks, player):
    return player_ranks[player][0] - STDEVS*player_ranks[player][1]

import random
def main(argv):
    games = []
    reader = csv.reader(open(argv[1]))
    for row in reader:
        scorelist = row[6:]
        assert len(scorelist) % 4 == 0
        players = []
        for playernum in xrange(len(scorelist) // 4):
            name, win, pts, turns = scorelist[playernum*4 : (playernum+1)*4]
            players.append((name, int(pts), int(turns)))
        
        # fixme: this awards ties to later players, which isn't actually
        # correct
        players.sort(key = lambda x: (x[1], -x[2]))
        for loser in xrange(len(players) - 1):
            for winner in xrange(loser + 1, len(players)):
                games.append( (players[winner][0], players[loser][0]) )

    player_ranks = {}
    
    for winner, loser in games:
        update(player_ranks, winner, loser)
    
    mus = []
    histogram = np.zeros((50,))
    print("Player ranks")
    print("============")
    for player in sorted(player_ranks, key=lambda player: rank(player_ranks, player)):
        mu = player_ranks[player][0]
        sigma = player_ranks[player][1]
        level = max(0, int(round(rank(player_ranks, player))))
        print("%20s   Lv %2d   Skill=%5.2f +- %5.2f" % (player, level, mu, sigma*STDEVS))
        mus.append(mu)
        bin = int(max(0, np.floor(rank(player_ranks, player))))
        histogram[bin] += 1
    print()
    print("Statistics")
    print("==========")
    print("mean =", np.mean(mus))
    print("stdev =", np.std(mus))
    print()

    total = np.sum(histogram)
    partials = np.cumsum(histogram)
    percentiles = partials / total
    for i in xrange(50):
        if histogram[i] > 0:
            print("Level %2d: %4d players, percentile %4.1f" % (i, histogram[i], percentiles[i]*100))

    win_levels = np.zeros((100,))
    loss_levels = np.zeros((100,))
    for winner, loser in games:
        upset = False
        diff = rank(player_ranks, winner) - rank(player_ranks, loser)
        if diff < 0:
            upset = True
            diff = -diff
        if upset:
            loss_levels[int(diff)] += 1
        else:
            win_levels[int(diff)] += 1
    probs = 1.0 - (win_levels+1) / (win_levels + loss_levels + 1)
    
    print()
    for diff in (1, 5, 10, 15, 20, 25, 30, 40):
        print("Probablity of %d-point upset: %4.1f%%" % (diff, probs[diff]*100))
    return player_ranks, probs

if __name__ == "__main__":
    main(sys.argv)
