# autograderP3.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
Autograder for PacPack
"""
from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
from game import Grid
from game import Configuration
from game import Agent
from game import reconstituteGrid
import sys, util, types, time, random, imp
import keyboardAgents

# If you change these, you won't affect the server, so you can't cheat
KILL_POINTS = 0
SONAR_NOISE_RANGE = 13 # Must be odd
SONAR_NOISE_VALUES = [i - (SONAR_NOISE_RANGE - 1)/2 for i in range(SONAR_NOISE_RANGE)]
SIGHT_RANGE = 5 # Manhattan distance
MIN_FOOD = 2
TOTAL_FOOD = 60

DUMP_FOOD_ON_DEATH = True # if we have the gameplay element that dumps dots on death

LAYOUT_SEED = [54, 416, 2435, 165, 715, 223, 802, 195, 5604, 397]
sanityScores = [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200]

# LAYOUT_SEED = [13, 13]

def noisyDistance(pos1, pos2):
  return int(util.manhattanDistance(pos1, pos2) + random.choice(SONAR_NOISE_VALUES))

###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

class GameState:
  """
  A GameState specifies the full game state, including the food,
  agent configurations and score changes.

  GameStates are used by the Game object to capture the actual state of the game and
  can be used by agents to reason about the game.

  Much of the information in a GameState is stored in a GameStateData object.  We
  strongly suggest that you access that data via the accessor methods below rather
  than referring to the GameStateData object directly.
  """

  ####################################################
  # Accessor methods: use these to access state data #
  ####################################################

  def getLegalActions( self, agentIndex=0 ):
    """
    Returns the legal actions for the agent specified.
    """
    return AgentRules.getLegalActions( self, agentIndex )

  def generateSuccessor( self, agentIndex, action):
    """
    Returns the successor state (a GameState object) after the specified agent takes the action.
    """
    # if agentIndex == 1: time.sleep(0.5) # uncomment this line to show that time is working correctly
    # Copy current state
    state = GameState(self)
    #print("generating succ", agentIndex, state.mostRecentBroadcast)

    # Find appropriate rules for the agent
    AgentRules.applyAction( state, action, agentIndex )
    AgentRules.checkDeath(state, agentIndex)
    #AgentRules.decrementTimer(state.data.agentStates[agentIndex])

    # Book keeping
    state.data._agentMoved = agentIndex
    state.data.score += state.data.scoreChange
    state.data.timeleft = self.data.timeleft - 1
    return state

  def getAgentState(self, index):
    return self.data.agentStates[index]

  def getAgentPosition(self, index):
    """
    Returns a location tuple if the agent with the given index is observable;
    if the agent is unobservable, returns None.
    """
    agentState = self.data.agentStates[index]
    ret = agentState.getPosition()
    if ret:
      return tuple(int(x) for x in ret)
    return ret

  def getNumAgents( self ):
    return len( self.data.agentStates )

  def getScore( self ):
    """
    Returns a number corresponding to the current score.
    """
    return self.data.score

  def getFood(self):
    """
    Returns a matrix of food that corresponds to the food on the board.
    For the matrix m, m[x][y]=true if there is food in (x,y) that can
    be eaten by the pacman team.
    """
    return self.data.food

  def getWalls(self):
    """
    Just like getFood but for walls
    """
    return self.data.layout.walls

  def hasFood(self, x, y):
    """
    Returns true if the location (x,y) has food
    """
    return self.data.food[x][y]

  def hasWall(self, x, y):
    """
    Returns true if (x,y) has a wall, false otherwise.
    """
    return self.data.layout.walls[x][y]

  def isOver( self ):
    return self.data._isOver

  def getPacmanTeamIndices(self):
    """
    Returns a list of agent index numbers for the agents on the pacman team.
    """
    return self.pacmanTeam[:]

  def getGhostTeamIndices(self):
    """
    Returns a list of the agent index numbers for the agents on the ghost team.
    """
    return self.ghostTeam[:]

  def isOnPacmanTeam(self, agentIndex):
    """
    Returns true if the agent is a in the pacman team.
    """
    if self.teams[agentIndex]:
      return True
    else:
      return False

  def getAgentDistances(self):
    """
    Returns a noisy distance to each agent.
    """
    if 'agentDistances' in dir(self) :
      return self.agentDistances
    else:
      return None

  def getDistanceProb(self, trueDistance, noisyDistance):
    "Returns the probability of a noisy distance given the true distance"
    if noisyDistance - trueDistance in SONAR_NOISE_VALUES:
      return 1.0/SONAR_NOISE_RANGE
    else:
      return 0

  def getInitialAgentPosition(self, agentIndex):
    "Returns the initial position of an agent."
    return self.data.layout.agentPositions[agentIndex][1]

  #############################################
  #             Helper methods:               #
  # You shouldn't need to call these directly #
  #############################################

  def __init__( self, prevState = None ):
    """
    Generates a new state by copying information from its predecessor.
    """
    if prevState != None: # Not initial state
      self.data = GameStateData(prevState.data)
      self.ghostTeam = prevState.ghostTeam
      self.pacmanTeam = prevState.pacmanTeam
      self.data.timeleft = prevState.data.timeleft

      self.teams = prevState.teams
      self.agentDistances = prevState.agentDistances
      #print("prev broad", prevState.mostRecentBroadcast)
      self.mostRecentBroadcast = prevState.mostRecentBroadcast
    else: # Initial state
      self.data = GameStateData()
      self.agentDistances = []
      self.mostRecentBroadcast = []

  def deepCopy( self ):
    state = GameState( self )
    state.data = self.data.deepCopy()
    state.data.timeleft = self.data.timeleft

    state.ghostTeam = self.ghostTeam[:]
    state.pacmanTeam = self.pacmanTeam[:]
    state.teams = self.teams[:]
    state.agentDistances = self.agentDistances[:]
    state.mostRecentBroadcast = self.mostRecentBroadcast
    return state

  def makeObservation(self, index):
    state = self.deepCopy()

    # Adds the sonar signal
    pos = state.getAgentPosition(index)
    n = state.getNumAgents()
    distances = [noisyDistance(pos, state.getAgentPosition(i)) for i in range(n)]
    state.agentDistances = distances

    # Remove states of distant opponents
    if index in self.ghostTeam:
      team = self.ghostTeam
      otherTeam = self.pacmanTeam
    else:
      otherTeam = self.ghostTeam
      team = self.pacmanTeam

    for enemy in otherTeam:
      seen = False
      enemyPos = state.getAgentPosition(enemy)
      for teammate in team:
        if util.manhattanDistance(enemyPos, state.getAgentPosition(teammate)) <= SIGHT_RANGE:
          seen = True
      # Got you Fixed the issue
      # if not seen: state.data.agentStates[enemy].configuration = None
    return state

  def __eq__( self, other ):
    """
    Allows two states to be compared.
    """
    if other == None: return False
    return self.data == other.data

  def __hash__( self ):
    """
    Allows states to be keys of dictionaries.
    """
    return int(hash( self.data ))

  def __str__( self ):

    return str(self.data)

  def initialize( self, layout, numAgents):
    """
    Creates an initial game state from a layout array (see layout.py).
    """
    self.data.initialize(layout, numAgents)
    positions = [a.configuration for a in self.data.agentStates]
    self.ghostTeam = [i for i,p in enumerate(positions) if not self.isPacman(p)]
    self.pacmanTeam = [i for i,p in enumerate(positions) if self.isPacman(p)]
    self.teams = [self.isPacman(p) for p in positions]
    #This is usually 60 (always 60 with random maps)
    #However, if layout map is specified otherwise, it could be less
    global TOTAL_FOOD
    TOTAL_FOOD = layout.totalFood

  def isPacman(self, configOrPos):
    """
    Returns whether the passed in position is the starting
    point of a pacman or a ghost. The convention is that pacman 
    agents start on the left and ghosts on the right side 
    of the board.
    """
    width = self.data.layout.width
    if type(configOrPos) == type( (0,0) ):
      return configOrPos[0] < width / 2
    else:
      return configOrPos.pos[0] < width / 2

############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                                                                          #
# You shouldn't need to look through the code in this section of the file. #
############################################################################

COLLISION_TOLERANCE = 0.7 # How close ghosts must be to Pacman to kill

class CaptureRules:
  """
  These game rules manage the control flow of a game, deciding when
  and how the game starts and ends.
  """

  def __init__(self, quiet = False):
    self.quiet = quiet

  def newGame( self, layout, agents, display, length, muteAgents, catchExceptions ):
    initState = GameState()
    initState.initialize( layout, len(agents) )
    starter = random.randint(0,1)
    # TODO: fix "ghost team starts" if there are no ghosts
    print(('%s team starts' % ['Pacman', 'Ghost'][starter]))
    game = Game(agents, display, self, startingIndex=starter, muteAgents=muteAgents, catchExceptions=catchExceptions)
    game.state = initState
    game.length = length
    game.state.data.timeleft = length
    if 'drawCenterLine' in dir(display):
      display.drawCenterLine()
    self._initFood = initState.getFood().count()
    return game

  def process(self, state, game):
    """
    Checks to see whether it is time to end the game.
    """
    # Check if there is any time left
    if state.data.timeleft <= 0:
      state.data._isOver = True

    if state.isOver():
      game.gameOver = True
      if not game.rules.quiet:
        foodToWin = TOTAL_FOOD - MIN_FOOD
        
        food_remaining = state.getFood().count()
        food_eaten = TOTAL_FOOD - food_remaining
        if food_remaining <= MIN_FOOD: # PackPac has eaten all the dots its supposed to
          print('The Pacman team has eaten all dots (except %d) in %d moves' % (MIN_FOOD, game.length - state.data.timeleft))
        elif state.data.timeleft == 0:
          print('Time is up.')
          if state.data.score == 0: print('The Pacman team ate no pellets! Sad!')
          else:
            print('The Pacman team has eaten %d pellets out of %d necessary for a complete success!' % (food_eaten, TOTAL_FOOD - MIN_FOOD))
        else:
          print('Game ended for an unknown reason.')

  def agentCrash(self, game, agentIndex):
    if agentIndex % 2 == 0:
      print("Pacman with index {} crashed".format(agentIndex), file=sys.stderr)
    else:
      print("Ghost with index {} crashed".format(agentIndex), file=sys.stderr)

  def getMaxTotalTime(self, agentIndex):
    return 900  # Move limits should prevent this from ever happening

  def getMaxStartupTime(self, agentIndex):
    return 15 # 15 seconds for registerInitialState

  def getMoveWarningTime(self, agentIndex):
    return 1  # One second per move

  def getMoveTimeout(self, agentIndex):
    return 3  # Three seconds results in instant forfeit

  def getMaxTimeWarnings(self, agentIndex):
    return 2  # Third violation loses the game

class AgentRules:
  """
  These functions govern how each agent interacts with her environment.
  """

  def getLegalActions( state, agentIndex ):
    """
    Returns a list of legal actions (which are both possible & allowed)
    """
    agentState = state.getAgentState(agentIndex)
    conf = agentState.configuration
    possibleActions = Actions.getPossibleActions( conf, state.data.layout.walls )
    return AgentRules.filterForAllowedActions( agentState, possibleActions)
  getLegalActions = staticmethod( getLegalActions )

  def filterForAllowedActions(agentState, possibleActions):
    return possibleActions
  filterForAllowedActions = staticmethod( filterForAllowedActions )


  def applyAction( state, action, agentIndex ):
    """
    Edits the state to reflect the results of the action.
    """
    legal = AgentRules.getLegalActions( state, agentIndex )
    if action not in legal:
      raise Exception("Illegal action " + str(action))

    # Update Configuration
    agentState = state.data.agentStates[agentIndex]
    speed = 1.0
    # if agentState.isPacman: speed = 0.5
    vector = Actions.directionToVector( action, speed )
    oldConfig = agentState.configuration
    agentState.configuration = oldConfig.generateSuccessor( vector )

    # Eat
    next = agentState.configuration.getPosition()
    nearest = nearestPoint( next )

    if next == nearest:
      isPacman = state.isOnPacmanTeam(agentIndex)
      # Change agent type
      agentState.isPacman = isPacman

    if agentState.isPacman and manhattanDistance( nearest, next ) <= 0.9 :
      AgentRules.consume( nearest, state, state.isOnPacmanTeam(agentIndex) )

  applyAction = staticmethod( applyAction )

  def consume( position, state, isPacman ):
    x,y = position
    # Eat food
    if state.data.food[x][y]:

      # Score increase when one pellet is eaten
      score = 1

      # do all the score and food grid maintainenace 
      state.data.scoreChange += score
      state.data.food = state.data.food.copy()
      state.data.food[x][y] = False
      state.data._foodEaten = position
      
      # Set state to win
      if (state.getFood().count() == MIN_FOOD):
       state.data._isOver = True

  consume = staticmethod( consume )

  def checkDeath( state, agentIndex):
    agentState = state.data.agentStates[agentIndex]
    if state.isOnPacmanTeam(agentIndex):
      otherTeam = state.getGhostTeamIndices()
    else:
      otherTeam = state.getPacmanTeamIndices()
    if agentState.isPacman:
      for index in otherTeam:
        otherAgentState = state.data.agentStates[index]
        if otherAgentState.isPacman: continue
        ghostPosition = otherAgentState.getPosition()
        if ghostPosition == None: continue
        if manhattanDistance( ghostPosition, agentState.getPosition() ) <= COLLISION_TOLERANCE:
          # award points to the other team for killing Pacmen
          score = KILL_POINTS
          if state.isOnPacmanTeam(agentIndex):
            score = -score
          state.data.scoreChange += score
          state.data.num_deaths += 1
          agentState.isPacman = True
          agentState.configuration = agentState.start
          
    else: # Agent is a ghost
      for index in otherTeam:
        otherAgentState = state.data.agentStates[index]
        if not otherAgentState.isPacman: continue
        pacPos = otherAgentState.getPosition()
        if pacPos == None: continue
        if manhattanDistance( pacPos, agentState.getPosition() ) <= COLLISION_TOLERANCE:
          # award points to the other team for killing Pacmen
          score = KILL_POINTS
          if not state.isOnPacmanTeam(agentIndex):
            score = -score
          state.data.scoreChange += score
          state.data.num_deaths += 1
          otherAgentState.isPacman = False
          otherAgentState.configuration = otherAgentState.start
          
  checkDeath = staticmethod( checkDeath )

  def placeGhost(state, ghostState):
    ghostState.configuration = ghostState.start
  placeGhost = staticmethod( placeGhost )

#############################
# FRAMEWORK TO START A GAME #
#############################

def default(str):
  return str + ' [Default: %default]'

def parseAgentArgs(str):
  if str == None or str == '': return {}
  pieces = str.split(',')
  opts = {}
  for p in pieces:
    if '=' in p:
      key, val = p.split('=')
    else:
      key,val = p, 1
    opts[key] = val
  return opts

def readCommand( argv ):
  """
  Processes the command used to run pacman from the command line.
  """
  from optparse import OptionParser
  usageStr = """
  USAGE:      python pacman.py <options>
  EXAMPLES:   (1) python capture.py
                  - starts a game with two baseline agents
              (2) python capture.py --keys0
                  - starts a two-player interactive game where the arrow keys control agent 0, and all other agents are baseline agents
              (3) python capture.py -p baselineTeam -b myTeam
                  - starts a fully automated game where the pacman team is a baseline team and ghost team is myTeam 
  """
  # TODO: Update above according to final defaults
  parser = OptionParser(usageStr)
  parser.add_option('-p', '--pacman', help=default('Pacman team'),
                    default='team') # TODO: Think about if we should leave this default
  parser.add_option('-g', '--ghost', help=default('Ghost team'),
                    default='oneGhostTeam')
  parser.add_option('--pacman-name', help=default('Pacman team name'),
                    default='Pacman')
  parser.add_option('--ghost-name', help=default('Ghost team name'),
                    default='Ghost')
  parser.add_option('--keys0', help='Make agent 0 (first pacman player) a keyboard agent', action='store_true',default=False)
  parser.add_option('--keys1', help='Make agent 1 (second pacman player) a keyboard agent', action='store_true',default=False)
  parser.add_option('--keys2', help='Make agent 2 (first ghost player) a keyboard agent', action='store_true',default=False)
  parser.add_option('--keys3', help='Make agent 3 (second ghost player) a keyboard agent', action='store_true',default=False)
  parser.add_option('-l', '--layout', dest='layout',
                    help=default('the LAYOUT_FILE from which to load the map layout; use RANDOM for a random maze; use RANDOM<seed> to use a specified random seed, e.g., RANDOM23'),
                    metavar='LAYOUT_FILE', default='defaultCapture')
  parser.add_option('-t', '--textgraphics', action='store_true', dest='textgraphics',
                    help='Display output as text only', default=False)

  parser.add_option('-q', '--quiet', action='store_true',
                    help='Display minimal output and no graphics', default=False)

  parser.add_option('-Q', '--super-quiet', action='store_true', dest="super_quiet",
                    help='Same as -q but agent output is also suppressed', default=False)

  parser.add_option('-z', '--zoom', type='float', dest='zoom',
                    help=default('Zoom in the graphics'), default=1)
  parser.add_option('-i', '--time', type='int', dest='time',
                    help=default('TIME limit of a game in moves'), default=1200, metavar='TIME')
  parser.add_option('-n', '--numGames', type='int',
                    help=default('Number of games to play'), default=10)
  parser.add_option('-f', '--fixRandomSeed', action='store_true',
                    help='Fixes the random seed to always play the same game', default=False)
  parser.add_option('--record', action='store_true',
                    help='Writes game histories to a file (named by the time they were played)', default=False)
  parser.add_option('--replay', default=None,
                    help='Replays a recorded game file.')
  # TODO: This currently doesn't work, consider removing or fixing
  parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                    help=default('How many episodes are training (suppresses output)'), default=0) 
  parser.add_option('-c', '--catchExceptions', action='store_true', default=True,
                    help='Catch exceptions and enforce time limits')

  options, otherjunk = parser.parse_args(argv)
  assert len(otherjunk) == 0, "Unrecognized options: " + str(otherjunk)
  args = dict()

  # Choose a display format
  if options.textgraphics:
    import textDisplay
    args['display'] = textDisplay.PacmanGraphics()
  elif options.quiet:
    import textDisplay
    args['display'] = textDisplay.NullGraphics()
  elif options.super_quiet:
    import textDisplay
    args['display'] = textDisplay.NullGraphics()
    args['muteAgents'] = True
  else:
    import captureGraphicsDisplay
    # Hack for agents writing to the display
    captureGraphicsDisplay.FRAME_TIME = 0
    args['display'] = captureGraphicsDisplay.PacmanGraphics(options.pacman, options.ghost, options.zoom, 0, capture=True)
    import __main__
    __main__.__dict__['_display'] = args['display']


  args['pacmanTeamName'] = options.pacman_name
  args['ghostTeamName'] = options.ghost_name

  if options.fixRandomSeed: random.seed('cs188')

  # Special case: recorded games don't use the runGames method or args structure
  if options.replay != None:
    print('Replaying recorded game %s.' % options.replay)
    import pickle
    recorded = pickle.load(open(options.replay))
    recorded['display'] = args['display']
    replayGame(**recorded)
    sys.exit(0)

  # Choose a pacman agent
  nokeyboard = options.textgraphics or options.quiet
  pacmanAgents = loadAgents(True, 'team', nokeyboard, {})
  pacmanSelfAgents = loadAgents(True, 'team_self', nokeyboard, {})
  ghostAgents = loadAgents(False, 'oneGhostTeam', nokeyboard, {})
  ghostSelfAgents = loadAgents(False, 'oneGhostTeam', nokeyboard, {})

  # Assume 2 agents on the pacman side, and
  # variable amount (0-2) on the ghost side
  args['agents'] = pacmanAgents + ghostAgents
  args['agents_self'] = pacmanSelfAgents + ghostSelfAgents

  numKeyboardAgents = 0
  for index, val in enumerate([options.keys0, options.keys1, options.keys2, options.keys3]):
    if not val: continue
    if numKeyboardAgents == 0:
      agent = keyboardAgents.KeyboardAgent(index)
    elif numKeyboardAgents == 1:
      agent = keyboardAgents.KeyboardAgent2(index)
    else:
      raise Exception('Max of two keyboard agents supported')
    numKeyboardAgents += 1
    args['agents'][index] = agent

  # Generate the layouts
  args['layouts'] = generateLayouts(LAYOUT_SEED, ghostAgents)




  args['length'] = options.time
  args['numGames'] = options.numGames
  args['numTraining'] = options.numTraining
  args['record'] = options.record
  args['catchExceptions'] = options.catchExceptions
  return args

def randomLayout(seed = None):
  if not seed:
    seed = random.randint(0,99999999)
  # layout = 'layouts/random%08dCapture.lay' % seed
  import mazeGenerator
  return mazeGenerator.generateMaze(seed)

def generateLayouts(seedLst, ghostAgents):
  import layout
  return [layout.Layout(randomLayout(seed).split('\n'), maxGhosts=len(ghostAgents)) for seed in seedLst]

import traceback
def loadAgents(isPacmanTeam, factory, textgraphics, cmdLineArgs):
  "Calls agent factories and returns lists of agents"
  try:
    if factory == "None":
      return []

    if not factory.endswith(".py"):
      factory += ".py"

    module = imp.load_source('player' + str(int(isPacmanTeam)), factory)
  except (NameError, ImportError):
    print('Error: The team "' + factory + '" could not be loaded! ', file=sys.stderr)
    traceback.print_exc()
    return [None for i in range(2)]
  except IOError as e:
    raise IOError('Error: The team "' + factory + '" could not be loaded! Check that the file exists!')

  args = dict()
  args.update(cmdLineArgs)  # Add command line args with priority

  print("Loading Team:", factory)
  print("Arguments:", args)

  # if textgraphics and factoryClassName.startswith('Keyboard'):
  #   raise Exception('Using the keyboard requires graphics (no text display, quiet or training games)')

  try:
    createTeamFunc = getattr(module, 'createTeam')
  except AttributeError:
    print('Error: The team "' + factory + '" could not be loaded! ', file=sys.stderr)
    traceback.print_exc()
    return [None for i in range(2)]

  indexAddend = 0
  if not isPacmanTeam:
    indexAddend = 2

  indices = [i + indexAddend for i in range(2)]
  return createTeamFunc(indices[0], indices[1], isPacmanTeam, **args)


def runGames(layouts, agents, agents_self, display, length, numGames, record, numTraining, pacmanTeamName, ghostTeamName, muteAgents=False, catchExceptions=True ):

  rules = CaptureRules()
  games = []

  if numTraining > 0:
    print('Playing %d training games' % numTraining)
  
  for i, l in enumerate(layouts):
    beQuiet = i < numTraining
    if beQuiet:
        # Suppress output and graphics
        import textDisplay
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        gameDisplay = display
        rules.quiet = False
    print("Running game with -l RANDOM%d  "%LAYOUT_SEED[i])
    g = rules.newGame( l, agents, gameDisplay, length, muteAgents, catchExceptions )
    g.run()
    print("")
    if not beQuiet: games.append(g)
  
  print("")
  #print "Preliminary Phase 3 Autograder Results:"
  print("*Results for sanity test against the staffbot: your agent should finish most games without crashing or timing out (time taken < 1200)*")
  passedCount = 0
  points = 0
  # Output
  print("Regular order")
  for i, game in enumerate(games): 
    if game.agentCrashed:
      myTime = 99999999
    else:
      myTime = game.length - game.state.data.timeleft

    currPassed = True
    targetTime = sanityScores[i]
    if myTime <= targetTime:
      passedCount += 1
    else:
      currPassed = False
    print("Game {}: -l RANDOM{} \tTime taken: {} \tTarget time: {} \t {} \t{}".format(i, LAYOUT_SEED[i], myTime, targetTime, "AGENT CRASHED" if game.agentCrashed else "", "" if currPassed else " (NOT PASSED)"))

  print("Sanity Test Summary: ", passedCount, "/", len(LAYOUT_SEED), "passed sanity test")
  print("Sanity Test Score: ", (passedCount / float(len(LAYOUT_SEED))), "/", "1.0")

  # if currPassed == len(LAYOUT_SEED):
  #   points += 1
  #   print "Passed sanity test with staffbot, received 1/1 on this part"

  
  # Self Team Testing (where you pair up with yourself)
  games_self = []
  for i, l in enumerate(layouts):
    beQuiet = i < numTraining
    if beQuiet:
        # Suppress output and graphics
        import textDisplay
        gameDisplay = textDisplay.NullGraphics()
        rules.quiet = True
    else:
        gameDisplay = display
        rules.quiet = False
    print("Running game with -l RANDOM%d  "%LAYOUT_SEED[i])
    g = rules.newGame( l, agents_self, gameDisplay, length, muteAgents, catchExceptions )
    g.run()
    print("")
    if not beQuiet: games_self.append(g)

  print("")
  print("*Results for self-play test, where your agent cooperates against a version of itself*")
  passedCount_self = 0
  myTimes_self = []
  for i, game in enumerate(games_self): 
    if game.agentCrashed:
      myTime = 99999999
    else:
      myTime = game.length - game.state.data.timeleft

    myTimes_self.append(myTime)
    currPassed = True
    print("Game {}: -l RANDOM{} \tTime taken: {} \t {} \t{}".format(i, LAYOUT_SEED[i], myTime, "AGENT CRASHED" if game.agentCrashed else "", "" if currPassed else " (NOT PASSED)"))

  bestEight = sorted(myTimes_self)[:8]
  bestEightAvg = sum(bestEight)/8.0
  print("Self-Play Summary: Average time for best 8 games is {}".format(bestEightAvg))
  return games + games_self








def save_score(game):
    with open('score', 'w') as f:
        print(game.state.data.score, file=f)

if __name__ == '__main__':
  """
  The main function called when pacman.py is run
  from the command line:

  > python capture.py

  See the usage string for more details.

  > python capture.py --help
  """
  options = readCommand( sys.argv[1:] ) # Get game components based on input
  games = runGames(**options)

  save_score(games[0])
  # import cProfile
  # cProfile.run('runGames( **options )', 'profile')
