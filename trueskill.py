from math import *
from math import pi as π
from pprint import pprint

games = [
         ("PK", "SF"), ("RT", "XT"), ("ACE", "BS"), ("SRM", "GB"), #day 1
         ("SysX", "XT"), ("PK", "RT"), ("SF", "GB"), ("SRM", "BS"), #day 2
         ("SysX", "GB"), ("RT", "BS"), ("SF", "SRM"), ("XT", "ACE"), #day 3
         ("SysX", "PK"), ("RT", "SF"), ("XT", "BS"), ("GB", "SRM"), #day 4
         ("SysX", "RT"), ("PK", "ACE"), ("XT", "SF"), ("BS", "GB"), #day 5
         ("SysX", "SF"), ("RT", "GB"), ("XT", "PK"), ("ACE", "SRM"), #day 6
         ("SysX", "ACE"), ("RT", "SF"), ("XT", "PK"), ("BS", "SRM"), #day 7
         ("SysX", "BS"), ("PK", "GB"), ("RT", "SRM"), ("ACE", "SF"), #day 8
         ("SysX", "RT"), ("PK", "SRM"), ("XT", "SF"), ("BS", "ACE"), #day 9
         ("PK", "SF"), ("XT", "SysX"), ("SRM", "ACE"), ("BS", "GB"), #day 10
         ("SysX", "SF"), ("PK", "RT"), ("XT", "SRM"), ("ACE", "GB"), #day 11
         ("SysX", "SRM"), ("PK", "BS"), ("RT", "ACE"), ("XT", "GB"), #day 12
         ("SysX", "PK"), ("RT", "XT"), ("BS", "ACE"), ("GB", "ACE"), #day 13
        ]

#http://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python/457805#457805
def erf(x):
    # save the sign of x
    sign = 1
    if x < 0: 
        sign = -1
    x = abs(x)
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x)
    return sign*y

e = exp(1)
pdf = lambda x: 1/(2*π)**.5 * e**(-x**2/2)
cdf = lambda x: (1 + erf(x/π**.5))/2

β = 25/6
γ = 25/300
ε = 0.08

Vwin = lambda t,e: pdf(t-e)/cdf(t-e)
Wwin = lambda t,e: Vwin(t,e)*(Vwin(t,e)+t-e)

def true_skill(winner, loser):

  µw, σw = winner
  µl, σl = loser

  c = (2*β**2 + σw**2 + σl**2)**.5
  t = (µw-µl)/c
  e = ε/c
  
  σw_new = (σw**2 * (1 - (σw**2)/(c**2)*Wwin(t, e)) + γ**2)**.5
  σl_new = (σl**2 * (1 - (σl**2)/(c**2)*Wwin(t, e)) + γ**2)**.5
  µw_new = (µw + σw**2/c * Vwin(t,e))
  µl_new = (µl - σl**2/c * Vwin(t,e))
  
  winner = µw_new, σw_new
  loser = µl_new, σl_new
  
  return winner, loser
  
def update(winner, loser):
  winner_stats = squads.get(winner, (25, 25/3))
  loser_stats = squads.get(loser, (25, 25/3))
  squads[winner], squads[loser] = true_skill(winner_stats, loser_stats)

rank = lambda x: squads[x][0] - 3*squads[x][1]

if __name__ == "__main__":
  
  squads = {}
  
  for game in games:
    update(*game)
  
  for squad in sorted(squads, key=rank):
    µ = squads[squad][0]
    σ = squads[squad][1]
    print("%6s R: %5.2f µ: %5.2f σ: %5.2f" % (squad, rank(squad), µ, σ))