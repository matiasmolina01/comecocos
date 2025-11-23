# my_team.py — Optimized Reflex Agents
# Compatible with UPF / Berkeley Capture the Flag contest

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

############################
# Team Creation
############################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]


############################
# Base Reflex Agent
############################

class ReflexCaptureAgent(CaptureAgent):
    """Adding helper utilities shared by both agents."""

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def closest(self, pos, targets):
        return min(self.manhattan(pos, t) for t in targets) if targets else 0


############################
# Offensive Agent (Optimized)
############################

class OffensiveReflexAgent(ReflexCaptureAgent):

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        carrying = my_state.num_carrying

        # --- FOOD FEATURE ---
        food_list = self.get_food(successor).as_list()
        features['food_left'] = -len(food_list)

        # distance to nearest food
        if food_list:
            distances = [self.get_maze_distance(my_pos, f) for f in food_list]
            features['dist_to_food'] = min(distances)

        # --- GHOST DANGER ---
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if (not e.is_pacman) and e.get_position() is not None]
        
        min_ghost_dist = 9999
        for g in ghosts:
            dist = self.get_maze_distance(my_pos, g.get_position())
            if dist < min_ghost_dist:
                min_ghost_dist = dist

        features['closest_ghost'] = min_ghost_dist

        # --- RETURN HOME LOGIC ---
        return_home = 0
        if carrying >= 2 or min_ghost_dist <= 4:
            return_home = 1

        features['return_home'] = return_home

        # distance to home (start location)
        home_dist = self.get_maze_distance(my_pos, self.start)
        features['dist_home'] = home_dist

        return features
    def get_weights(self, game_state, action):
        return {
            'food_left': 100,
            'dist_to_food': -1,
            'closest_ghost': 2,
            'return_home': 150,   # 🟢 FUERZA A VOLVER
            'dist_home': -1       # 🟢 ACERCA MÁS RÁPIDO A CASA
        }



############################
# Defensive Agent (Optimized)
############################

class DefensiveAgent(ReflexCaptureAgent):

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        actions = [a for a in actions if a != Directions.STOP]

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights()
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # 1. On defense?
        features['on_defense'] = 0 if my_state.is_pacman else 1

        # 2. Detect invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position()]

        features['num_invaders'] = len(invaders)

        if invaders:
            dists = [self.manhattan(my_pos, inv.get_position()) for inv in invaders]
            features['invader_distance'] = min(dists)

        # 3. Patrol near important food
        defending_food = self.get_food_you_are_defending(successor).as_list()
        if defending_food:
            # pick the food closest to our start (chokepoint-like)
            important_food = min(defending_food, key=lambda x: self.manhattan(self.start, x))
            features['distance_to_important_food'] = self.manhattan(my_pos, important_food)

        # 4. Avoid STOP or REVERSE
        if action == Directions.STOP:
            features['stop'] = 1
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    def get_weights(self):
        return {
            'num_invaders': -1000,
            'invader_distance': -10,
            'on_defense': 100,
            'distance_to_important_food': -2,
            'stop': -100,
            'reverse': -2
        }
