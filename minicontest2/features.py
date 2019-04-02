# for offensive agent
def getFeatures(self, gameState, action):
    # return a counter of features for this state
    successor = self.getSuccessor(gameState, action)
    old_state = self.getPreviousObservation()
    eat_food = self.getFood(successor)
    food_list = eat_food.asList()
    defend_food = self.getFoodYouAreDefending(successor)
    opponents_index = self.getOpponents(successor)
    walls = successor.getWalls()
    features = util.Counter()
    features['bias'] = 1.0
    # compute the location of pacman after the action
    new_state = successor.getAgentState(self.index)
    next_x, next_y = new_state.getPosition()
    # calculate distance to opponents
    oppo_position = [successor.getAgentState(oppo).getPosition() for oppo in\
                     opponents_index]
    distance_to_oppo = [self.getMazeDistance((next_x, next_y), oppo) for oppo\
                        in oppo_position]
    closest_distance = min(distance_to_oppo)
    avg_distance = sum(distance_to_oppo) / len(distance_to_oppo)
    features['closest_distance_to_ghost'] = closest_distance
    features['average_distance_to_ghost'] = avg_distance
    num_of_ghost = 0
    for distance_oppo in distance_to_oppo:
        if distance_oppo < 3:
            num_of_ghost += 1
    features['num_of_ghost_nearby'] = num_of_ghost
    is_surrounded = oppo_position[0][0] <= next_x <= oppo_position[1][0] or\
                    oppo_position[0][1] <= next_y <= oppo_position[1][1]
    features['is_surrounded_by_ghost'] = is_surrounded
    # closest to food
    if food_list:
        min_distance = min([self.getMazeDistance((next_x, next_y), food_list)\
                            for food in food_list])
        features['min_distance_to_food'] = min_distance
    get_food = eat_food[next_x][next_y]
    features['eat_food'] = get_food
    # carrying food
    last_role = old_state.getAgentState(self.index).isPacman()
    cur_role = gameState.getAgentState(self.index).isPacman()
    if last_role ^ cur_role:
        self.carry_food = 0
    elif get_food:
        self.carry_food += 1
    features['carry_food'] = self.carry_food
    # distance to frontier
    frontier_x = walls.width / 2
    return_dis = abs(frontier_x - next_x)
    features['return_distance'] = return_dis
    # tunnel/crossing/dead end/open
    features['is_dead_end'] = self.is_dead_end[(next_x, next_y)]
    features['is_tunnel'] = self.is_tunnel[(next_x, next_y)]
    features['is_crossing'] = self.is_crossing[(next_x, next_y)]
    features['is_open_area'] = self.is_open_area[(next_x, next_y)]
    # TODO: power capsules, scared time

# for defensive agent
def getFeaturesDefensive(self, gameState, action):
    # return a counter of feature for new state
    successor = self.getSuccessor(gameState, action)
    old_state = self.getPreviousObservation()
    eat_food = self.getFood(successor)
    defend_food = self.getFoodYouAreDefending(successor)
    food_list = defend_food.asList()
    opponents_index = self.getOpponents(successor)
    oppo_position = [self.getAgentState(oppo).getPosition() for oppo\
                     in opponents_index]
    walls = successor.getWalls()
    features = util.Counter()
    features['bias'] = 1.0
    # opponent in tunnel/crossing/openarea
    features['oppo_in_dead_end'] = self.is_dead_end[(oppo_position[0])] or\
                                   self.is_dead_end[(oppo_position[1])]
    features['oppo_in_tunnel'] = self.is_tunnel[(oppo_position[0])] or\
                                 self.is_tunnel[(oppo_position[1])]
    features['oppo_in_crossing'] = self.is_crossing[(oppo_position[0])] or\
                                   self.is_crossing[(oppo_position[1])]
    features['oppo_in_open_area'] = self.is_open_area[(oppo_position[0])] or\
                                    self.is_open_area[(oppo_position[1])]
    # opponent location relative to self
    teammate_index = (self.index + 2) % 4
    teammate_x, teammate_y = self.getAgentState(teammate_index).getPosition()
    for pos in oppo_position:
        surrounded_x = min(teammate_x, next_x) <= oppo_position[0][0] <=
                       max(teammate_x, next_x) or
                       min(teammate_x, next_x) <= oppo_position[1][0] <=
                       max(teammate_x, next_x)

        surrounded_y = min(teammate_y, next_y) <= oppo_position[0][1] <=
                       max(teammate_y, next_y) or
                       min(teammate_y, next_y) <= oppo_position[1][1] <=
                       max(teammate_y, next_y)
        surrounded_both = surrounded_x and surrounded_y
        features['surrounded_x'] = surrounded_x
        features['surrounded_y'] = surrounded_y
        features['surrounded_both'] = surrounded_both
    # distance to frontier
    frontier_x = walls.width / 2
    frontier_dis = abs(frontier_x - next_x)
    features['frontier_distance'] = return_dis
    # self in tunnel/crossing/dead end/open
    features['is_dead_end'] = self.is_dead_end[(next_x, next_y)]
    features['is_tunnel'] = self.is_tunnel[(next_x, next_y)]
    features['is_crossing'] = self.is_crossing[(next_x, next_y)]
    features['is_open_area'] = self.is_open_area[(next_x, next_y)]
    # self is scared
    features['is_scared'] = self.getAgentState(self.index).scaredTimer > 0
    # calculate distance to opponents
    distance_to_oppo = [self.getMazeDistance((next_x, next_y), oppo) for oppo\
                        in oppo_position]
    closest_distance = min(distance_to_oppo)
    avg_distance = sum(distance_to_oppo) / len(distance_to_oppo)
    features['closest_distance_to_pacman'] = closest_distance
    features['average_distance_to_pacman'] = avg_distance
    # calculate opponent distance to defending dots
    if food_list:
        min_distance = 1e9
        for oppo in oppo_position:
            # potential bug
            min_distance = min([self.getMazeDistance(oppo, food_list)\
                            for food in food_list], min_distance)
        features['min_distance_to_defend_food'] = min_distance
    # get dot loss
    if self.red:
        old_defending_food = old_state.getRedFood()
    else:
        old_defending_food = old_state.getBlueFood()
    dot_loss = 0
    for oppo in oppo_position:
        dot_loss += old_defending_food[int(oppo[0])][int(oppo[1])]
    features['dot_loss'] = dot_loss






