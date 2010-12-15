#!/usr/bin/env python2.6
from __future__ import print_function
import math
import sys
import csv

from scipy.stats.distributions import norm as scipy_norm
import numpy as np

beta = 21.
gamma = 1./12
epsilon = 0.5

norm = scipy_norm()

def pdf(x):
    return norm.pdf(x)

def cdf(x):
    return norm.cdf(x)

def Vwin(t, e):
    return pdf(t - e) / cdf(t - e)

def Wwin(t, e):
    return Vwin(t, e) * (Vwin(t, e) + t - e)

def Vdraw(t, e):
    return (pdf(-e - t) - pdf(e - t)) / (cdf(e - t) - cdf(-e - t))

def Wdraw(t, e):
    adjustment_num = (e - t)*pdf(e - t) + (e + t)*pdf(e + t)
    adjustment_denom = cdf(e - t) - cdf(-e - t)
    return Vdraw(t, e) ** 2 + adjustment_num / adjustment_denom

def adjust_scores(winner, loser, Vfunc, Wfunc, loss_polarity=-1):
    muw, sigmaw = winner
    mul, sigmal = loser
    
    c = (2*beta**2 + sigmaw**2 + sigmal**2)**.5
    t = (muw - mul) / c
    e = epsilon / c

    sigmaw2_adjust = 1 - (sigmaw**2) / (c**2)*Wfunc(t, e)
    sigmal2_adjust = 1 - (sigmal**2) / (c**2)*Wfunc(t, e)

    muw_adjust = sigmaw**2/c * Vfunc(t, e)
    mul_adjust = loss_polarity * sigmal**2/c * Vfunc(t, e)

    return muw_adjust, sigmaw2_adjust, mul_adjust, sigmal2_adjust

def adjust_for_win(winner, loser):
    return adjust_scores(winner, loser, Vwin, Wwin, -1)

def adjust_for_draw(p1, p2):
    return adjust_scores(p1, p2, Vdraw, Wdraw, 1)

def update(player_ranks, winner, loser):
    winner_stats = player_ranks.get(winner, (25, 25./3))
    loser_stats = player_ranks.get(loser, (25, 25./3))
    player_ranks[winner], player_ranks[loser] = true_skill(winner_stats, loser_stats)

def rank(player_ranks, player):
    return player_ranks[player][0] - 3*player_ranks[player][1]

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
        print("%20s   Lv %2d   Skill=%5.2f +- %5.2f" % (player, level, mu, sigma*3))
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
