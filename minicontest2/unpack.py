# unpack.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import os, pickle, sys

if len(sys.argv) != 3:
  print('Usage: %s stats_file team_name' % sys.argv[0])
  print('Unpacks the stats file of a server into a bunch of replay files.')
  if len(sys.argv) == 2:
    d = pickle.load(open(sys.argv[1]))
    print('Team names:', list(d.keys()))
  sys.exit(2)

d = pickle.load(open(sys.argv[1]))
user = sys.argv[2]
k = 0
print('Unpacking games for', user)
for g, w in d[user]['gameHistory']:
    k += 1
    t = {'layout': g.state.data.layout, 'agents': g.agents, 'actions': g.moveHistory, 'length': g.length}
    fname = 'replay_' + user + '_' + str(k)
    print('Game:', fname)
    pickle.dump(t,file(fname, 'w'))
