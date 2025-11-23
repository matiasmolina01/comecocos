# my_team.py
import random
import util
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################
def create_team(first_index, second_index, is_red,
                first='SmartOffensiveAgent', second='SmartDefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


##################
# Helper Methods #
##################
def within_border(game_state, pos, is_red):
    # Returns True if pos is on the agent's own side (approx via is_red)
    # Build a dummy config to use state's is_red
    from game import Configuration
    conf = Configuration(pos, 'North')
    return game_state.is_red(conf) == is_red


################
# Base Agent   #
################
class SmartAgent(CaptureAgent):
    """
    Base utilities for our agents.
    """

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        # threshold of pellets to carry before returning
        self.return_threshold = 4

    def get_successor(self, game_state, action):
        # same logic as baseline: ensure successor is a grid point
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # half step, advance one more
            return successor.generate_successor(self.index, action)
        return successor

    def visible_enemies(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible = [e for e in enemies if e.get_position() is not None]
        return visible

    def nearest_visible_enemy(self, game_state, consider_scared=False):
        enemies = self.visible_enemies(game_state)
        best = None
        best_d = None
        for e in enemies:
            if not consider_scared and e.scared_timer > 0:
                continue
            pos = e.get_position()
            d = self.get_maze_distance(game_state.get_agent_state(self.index).get_position(), pos)
            if best is None or d < best_d:
                best, best_d = e, d
        return best, best_d


###############
# Offense     #
###############
class SmartOffensiveAgent(SmartAgent):
    """
    Offensive agent that:
    - targets food clusters
    - returns early when carrying threshold
    - avoids non-scared ghosts
    """

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        # filter out 'Stop' unless no other choice
        legal = [a for a in actions if a != Directions.STOP]
        if not legal:
            legal = actions

        values = [self.evaluate(game_state, a) for a in legal]
        max_val = max(values)
        best = [a for a, v in zip(legal, values) if v == max_val]
        return random.choice(best)

    def evaluate(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        is_red = game_state.is_on_red_team(self.index)

        # 1) prefer states that *reduce* food on opponent side (progress)
        opponent_food = self.get_food(successor).as_list()
        features['food_left'] = len(opponent_food)  # fewer is better

        # 2) distance to nearest food (prefer smaller)
        if opponent_food:
            dists = [self.get_maze_distance(my_pos, f) for f in opponent_food]
            features['dist_food'] = min(dists)
        else:
            features['dist_food'] = 0

        # 3) cluster size near the chosen food (prefer clusters)
        # estimate cluster by counting foods within radius 3 of closest food
        if opponent_food:
            closest_food = min(opponent_food, key=lambda f: self.get_maze_distance(my_pos, f))
            cluster = sum(1 for f in opponent_food if self.get_maze_distance(closest_food, f) <= 3)
            features['cluster'] = cluster
        else:
            features['cluster'] = 0

        # 4) carrying pellets (encourage returning home when carrying many)
        carrying = my_state.num_carrying if hasattr(my_state, 'num_carrying') else 0
        features['carrying'] = carrying

        # 5) ghost avoidance: penalize positions that are close to non-scared ghosts
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghost_dists = []
        for e in enemies:
            pos = e.get_position()
            if pos is None:
                continue
            # if the enemy is currently a ghost (i.e., on our side), or if it's a ghost on the opponent side we still avoid when close
            if e.scared_timer <= 0:
                d = self.get_maze_distance(my_pos, pos)
                ghost_dists.append(d)
        if ghost_dists:
            min_g = min(ghost_dists)
        else:
            min_g = 9999
        features['ghost_dist'] = min_g

        # 6) distance to home (when returning) -> prefer smaller when carrying many
        # home defined as the start position or nearest home-side safe point
        dist_home = self.get_maze_distance(my_pos, self.start)
        features['dist_home'] = dist_home

        # Weights tuned to push desired behaviour
        weights = {
            'food_left': -200.0,   # large negative: fewer food is big progress (reduce)
            'dist_food': -1.5,     # go to food
            'cluster': 2.0,        # prefer clusters
            'carrying': 20.0,      # encourage having pellets (so we bring them)
            'ghost_dist': 3.0,     # bigger ghost_dist is good; we will invert in scoring below
            'dist_home': -2.0      # prefer close to home when returning
        }

        # Compute raw linear sum. For ghost_dist we want closer ghosts to penalize: use -1/min_g effect
        value = 0.0
        value += weights['food_left'] * features['food_left']
        value += weights['dist_food'] * features['dist_food']
        value += weights['cluster'] * features['cluster']
        value += weights['carrying'] * features['carrying']

        # ghost term: if very close ghost, heavy penalty
        if features['ghost_dist'] <= 2:
            value -= 1000  # immediate danger: avoid death
        else:
            # reward safe distance; use log-like shaping
            value += weights['ghost_dist'] * min(features['ghost_dist'], 20)

        # returning behavior: if carrying >= threshold then bias to go home
        if features['carrying'] >= self.return_threshold:
            value += weights['dist_home'] * features['dist_home'] * 5.0  # stronger home preference
        else:
            # encourage to get nearer to the food
            pass

        return value


###############
# Defense     #
###############
class SmartDefensiveAgent(SmartAgent):
    """
    Defensive agent:
    - prefers to stay on home side
    - patrols near midline / choke points
    - chases visible invaders
    """

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        legal = [a for a in actions if a != Directions.STOP]
        if not legal:
            legal = actions

        values = [self.evaluate(game_state, a) for a in legal]
        max_val = max(values)
        best = [a for a, v in zip(legal, values) if v == max_val]
        return random.choice(best)

    def evaluate(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        is_red = game_state.is_on_red_team(self.index)

        # 1) Are we on defense? (not pacman)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # 2) invaders visible: enemy pacmen on our side
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            dists = [self.get_maze_distance(my_pos, e.get_position()) for e in invaders]
            features['invader_dist'] = min(dists)
        else:
            features['invader_dist'] = 0

        # 3) patrol distance: prefer staying near the midline / start area when no invaders
        # We set a patrol target near start but slightly forward (towards middle)
        patrol_target = self.compute_patrol_target(game_state, is_red)
        features['patrol_dist'] = self.get_maze_distance(my_pos, patrol_target)

        # 4) discourage stopping and reversing direction
        features['stop'] = 1 if action == Directions.STOP else 0
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        features['reverse'] = 1 if action == rev else 0

        # weights
        weights = {
            'on_defense': 100.0,
            'num_invaders': -1000.0,
            'invader_dist': -20.0,
            'patrol_dist': -1.0,
            'stop': -100.0,
            'reverse': -2.0
        }

        value = 0.0
        value += weights['on_defense'] * features['on_defense']
        value += weights['num_invaders'] * features['num_invaders']

        if features['num_invaders'] > 0:
            # when invader present, heavily weight invader distance (chase)
            value += weights['invader_dist'] * features['invader_dist']
        else:
            # no invader, prefer patrol position
            value += weights['patrol_dist'] * features['patrol_dist']

        value += weights['stop'] * features['stop']
        value += weights['reverse'] * features['reverse']

        # Additional safety: if we are scared, deprioritize chasing
        if my_state.scared_timer > 0:
            value -= 200  # avoid risky chasing when scared

        return value

    def compute_patrol_target(self, game_state, is_red):
        # choose a patrol tile roughly on the home side near the middle column
        layout = game_state.data.layout
        width = layout.width
        height = layout.height
        # pick a x coordinate that is one column inside our half (towards mid)
        if is_red:
            target_x = max(1, (width // 2) - 2)
        else:
            target_x = min(width - 2, (width // 2) + 2)
        # choose center y
        target_y = height // 2
        # if that cell is a wall, nudge up/down
        for dy in range(0, height//2):
            for sign in (1, -1):
                ny = target_y + sign*dy
                if ny < 0 or ny >= height: continue
                if not layout.walls[target_x][ny]:
                    return (target_x, ny)
        # fallback to agent start
        return self.start
