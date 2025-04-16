# SYSTEM IMPORTS
from collections import defaultdict
from typing import List, Type, Dict, Tuple
from pprint import pprint
from tqdm import tqdm
import argparse as ap
import copy as cp
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.collections as pltcollections
import matplotlib.ticker as pltticker
import numpy as np
import os
import sys
# import scipy.linalg as scpl
import time

from collections import defaultdict


# make sure the directory of this file is on path so we can load other files it depends on
_cd_: str = os.path.abspath(os.path.dirname(__file__))
if _cd_ in sys.path:
    sys.path.append(_cd_)
del _cd_


CELL_WIDTH: int = 1


np.random.seed(12345)


PolicyType = Type["Policy"]
ViterAgentType = Type["ValueIterationAgent"]
PiterAgentType = Type["PolicyIterationAgent"]
DUEAgentType = Type["DirectUtilityEstimationAgent"]
PADPAgentType = Type["PassiveAdaptiveDynamicProgrammingAgent"]


# PYTHON PROJECT IMPORTS
from world import Action, Cell, TransitionModel, Grid


def make_rectangles(world: Grid,
                    axes: List[plt.Axes]
                    ) -> List[List[List[plt.Rectangle]]]:
    all_rectangles = list()
    for ax in axes:
        all_rectangles.append([[plt.Rectangle((cell.x, cell.y),
                                               CELL_WIDTH, CELL_WIDTH,
                                               color="white")
                                 for cell in row]
                                for row in world.grid])

    return all_rectangles

def visualize_boundaries(world: Grid,
                         axes: List[plt.Axes]
                         ) -> None:
    for ax in axes:
        for row in world.grid:
            for cell in row:
                bx, by = cell.x, cell.y
                tx, ty = cell.x + CELL_WIDTH, cell.y + CELL_WIDTH


                cell_border_color = "black"
                ax.plot([bx, bx], [by, ty], color=cell_border_color)
                ax.plot([tx, tx], [by, ty], color=cell_border_color)
                ax.plot([bx, tx], [by, by], color=cell_border_color)
                ax.plot([bx, tx], [ty, ty], color=cell_border_color)

    for ax in axes:
        for row in world.grid:
            for cell in row:
                bx, by = cell.x, cell.y
                tx, ty = cell.x + CELL_WIDTH, cell.y + CELL_WIDTH

                if cell.is_start_state or cell.is_terminal_state:
                    cell_border_color = "magenta"
                    if cell.is_start_state:
                        cell_border_color = "blue"
                    ax.plot([bx, bx], [by, ty], color=cell_border_color)
                    ax.plot([tx, tx], [by, ty], color=cell_border_color)
                    ax.plot([bx, tx], [by, by], color=cell_border_color)
                    ax.plot([bx, tx], [ty, ty], color=cell_border_color)

def visualize_rewards(world: Grid,
                      rewards_rectangles: List[List[plt.Rectangle]],
                      reward_ax: plt.Axes
                      ) -> None:
    reward_ax.texts.clear()

    # plot the rewards
    for cell_row, recs_row in zip(world.grid, rewards_rectangles):
        for cell, rec in zip(cell_row, recs_row):
            if cell.is_available:
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                reward_ax.annotate("{0:.2f}".format(world.reward_function.get_reward(cell)),
                                   (cx, cy), color="black",
                                   weight="bold", ha="center", va="center")


def visualize_policy(world: Grid,
                     policy: PolicyType,
                     policy_rectangles: List[List[plt.Rectangle]],
                     policy_ax: plt.Axes
                     ) -> None:

    policy_ax.texts.clear()

    for cell_row, recs_row in zip(world.grid, policy_rectangles):
        for cell, rec in zip(cell_row, recs_row):
            if cell.is_available and not cell.is_terminal_state:

                action = policy.recommend(cell)

                bx, by = cell.x, cell.y
                tx, ty = cell.x + CELL_WIDTH, cell.y + CELL_WIDTH

                asx, asy, dx, dy = 0, 0, 0, 0
                if action == Action.UP:
                    asx, asy = bx + CELL_WIDTH/2, ty - CELL_WIDTH/8
                    dx, dy = 0, -CELL_WIDTH*6/8
                elif action == Action.LEFT:             # works
                    asx, asy = bx + CELL_WIDTH/8, by + CELL_WIDTH/2
                    dx, dy = CELL_WIDTH*6/8, 0
                elif action == Action.RIGHT:
                    asx, asy = tx - CELL_WIDTH/8, by + CELL_WIDTH/2
                    dx, dy = -CELL_WIDTH*6/8, 0
                else: # Action.DOWN
                    asx, asy = bx + CELL_WIDTH/2, by + CELL_WIDTH/8
                    dx, dy = 0, CELL_WIDTH*6/8

                policy_ax.annotate("", xy=(asx, asy), xytext=(asx+dx, asy+dy), arrowprops=dict(arrowstyle="->"))

                """
                action_str = f"{action}".replace("Action.", "")[0]

                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                policy_ax.annotate(action_str,
                                   (cx, cy), color="black",
                                   weight="bold", ha="center", va="center")
                """
            else:
                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                policy_ax.annotate("{0:.2f}".format(world.reward_function.get_reward(cell)),
                                   (cx, cy), color="black",
                                   weight="bold", ha="center", va="center")

def visualize_utilities(world: Grid,
                        utilities: Dict[Cell, float],
                        utility_rectangles: List[List[plt.Rectangle]],
                        utility_ax: plt.Axes
                        ) -> None:
    utility_ax.texts.clear()

    for cell_row, recs_row in zip(world.grid, utility_rectangles):
        for cell, rec in zip(cell_row, recs_row):
            if cell.is_available:
                utility = utilities[cell]

                rx, ry = rec.get_xy()
                cx = rx + rec.get_width() / 2.0
                cy = ry + rec.get_height() / 2.0
                utility_ax.annotate("{0:.3f}".format(utility),
                                   (cx, cy), color="black",
                                   weight="bold", ha="center", va="center")



def visualize_frequency_heatmap(world: Grid,
                                num_samples: Dict[Cell, int],
                                num_trajectories: int,
                                heatmap_rectangles: List[List[plt.Rectangle]],
                                heatmap_ax: plt.Axes
                                ) -> None:
    # pprint(num_samples)
    for cell_row, rec_row in zip(world.grid, heatmap_rectangles):
        for cell, rec in zip(cell_row, rec_row):
            cell_fill_color = "red"
            if cell.is_available:
                alpha = 0
                if num_trajectories > 0:
                    alpha = min(1, num_samples[cell] / num_trajectories)
                # print(cell, alpha)
                rec.set(color=cell_fill_color, alpha=alpha)


class Policy(object):
    def __init__(self: PolicyType,
                 world: Grid
                 ) -> None:
        self.world = world
        self.policy_map: Dict[Cell, Action] = dict()
        for x in range(self.world.num_x_coords):
            for y in range(self.world.num_y_coords):

                cell = self.world.get_cell(x, y)

                if not cell.is_terminal_state and cell.is_available:
                    self.policy_map[self.world.get_cell(x, y)] = np.random.choice(list(Action))

        # was for testing
        # self.policy_map[self.world.get_cell(0, 1)] = Action.RIGHT

    def recommend(self: PolicyType,
                  cell: Cell
                  ) -> Action:
        return self.policy_map[cell]

    def get_utility_of_recommended_action(self: PolicyType,
                                          cell: Cell,
                                          utilities: Dict[Cell, float] 
                                          ) -> float:
        recommended_utility: float = self.world.reward_function.get_reward(cell)
        recommended_action: Action = self.recommend(cell)
        for c, prob in TransitionModel.get_transition_probs(self.world, cell, recommended_action).items():
            recommended_utility += self.world.gamma * prob * utilities[c]
        return recommended_utility

    def get_best_action(self: PolicyType,
                        cell: Cell,
                        utilities: Dict[Cell, float]
                        ) -> Tuple[Action, float]:
        best_action = None
        best_utility = -math.inf

        for a in Action:
            action_utility = self.world.reward_function.get_reward(cell)
            for c, prob in TransitionModel.get_transition_probs(self.world, cell, a).items():
                action_utility += self.world.gamma * prob * utilities[c]

            if action_utility > best_utility:
                best_utility = action_utility
                best_action = a
        return best_action, best_utility

    def recalculate(self: PolicyType,
                    utilities: Dict[Cell, float],
                    ) -> None:
        for x in range(self.world.num_x_coords):
            for y in range(self.world.num_y_coords):

                cell = self.world.get_cell(x, y)

                if not cell.is_terminal_state and cell.is_available:
                    best_action, _ = self.get_best_action(cell, utilities)

                    self.policy_map[cell] = best_action


class ValueIterationAgent(object):
    def __init__(self: ViterAgentType,
                 world: Grid,
                 refresh_period: float,
                 epsilon: float = 1e-9
                 ) -> None:
        self.world = world
        self.refresh_period = refresh_period
        self.epsilon = epsilon

        self.policy = Policy(self.world)
        self.utilities: Dict[Cell, float] = {cell: 0.0 for cell_row in self.world.grid
                                                       for cell in cell_row
                                                       if cell.is_available}

    def visualize(self: ViterAgentType,
                  figure: plt.Figure,
                  all_rectangles: List[List[List[plt.Rectangle]]],
                  all_axes: List[plt.Axes],
                  **kwargs
                  ) -> None:

        for rectangles in all_rectangles:
            for cell_row, rec_row in zip(self.world.grid, rectangles):
                for cell, rec in zip(cell_row, rec_row):
                    cell_fill_color = "white"
                    if not cell.is_available:
                        cell_fill_color = "black"

                    rec.set(color=cell_fill_color)

        (reward_ax, policy_ax, observed_ax, utility_ax) = all_axes
        (reward_recs, policy_recs, observed_recs, utility_recs) = all_rectangles

        visualize_rewards(self.world, reward_recs, reward_ax)
        visualize_policy(self.world, self.policy, policy_recs, policy_ax)
        visualize_utilities(self.world, self.utilities, utility_recs, utility_ax)

        figure.canvas.draw()
        figure.canvas.flush_events()

    def solve(self: ViterAgentType,
              figure: plt.Figure,
              all_rectangles: List[List[List[plt.Rectangle]]],
              all_axes: List[plt.Axes]
              ) -> None:
        # implement viter algorithm
        u: Dict[Cell, float] = {cell: 0.0 for cell_row in self.world.grid
                                          for cell in cell_row
                                          if cell.is_available}

        u_prime: Dict[Cell, float] = cp.copy(u)

        delta = math.inf

        while delta > self.epsilon * (1 - self.world.gamma) / self.world.gamma:
            u = cp.copy(u_prime)
            delta = 0.0

            for cell in u.keys():
                utility = self.world.reward_function.get_reward(cell)

                if not cell.is_terminal_state:
                    max_action_utility = -math.inf

                    for a in Action:
                        action_utility = 0.0
                        for c, prob in TransitionModel.get_transition_probs(self.world, cell, a).items():
                            action_utility += prob * u[c]

                        max_action_utility = max(max_action_utility, action_utility)

                    utility += self.world.gamma * max_action_utility

                u_prime[cell] = utility

                if abs(u_prime[cell] - u[cell]) > delta:
                    delta = abs(u_prime[cell] - u[cell])

            self.utilities = u
            # self.policy.recalculate(self.utilities)
            self.visualize(figure, all_rectangles, all_axes)

            time.sleep(self.refresh_period)

        self.policy.recalculate(self.utilities)
        self.visualize(figure, all_rectangles, all_axes)

    def play_fit(self: ViterAgentType,
                 figure: plt.Figure,
                 all_rectangles: List[List[List[plt.Rectangle]]],
                 all_axes: List[plt.Axes]
                 ) -> None:
        return
                 


class PolicyIterationAgent(object):
    def __init__(self: PiterAgentType,
                 world: Grid,
                 refresh_period: float,
                 k: int = 15
                 ) -> None:
        self.world = world
        self.refresh_period = refresh_period
        self.k = k

        self.policy = Policy(self.world)
        self.utilities: Dict[Cell, float] = {cell: 0.0 for cell_row in self.world.grid
                                                       for cell in cell_row
                                                       if cell.is_available}

        self.state_to_idx_map: Dict[cell, int] = dict()
        idx: int = 0
        for x in range(self.world.num_x_coords):
            for y in range(self.world.num_y_coords):
                cell = self.world.get_cell(x, y)
                if cell.is_available:
                    self.state_to_idx_map[cell] = idx
                    idx += 1
        self.num_available_states = len(self.state_to_idx_map)

    def visualize(self: PiterAgentType,
                  figure: plt.Figure,
                  all_rectangles: List[List[List[plt.Rectangle]]],
                  all_axes: List[plt.Axes],
                  **kwargs
                  ) -> None:

        for rectangles in all_rectangles:
            for cell_row, rec_row in zip(self.world.grid, rectangles):
                for cell, rec in zip(cell_row, rec_row):
                    cell_fill_color = "white"
                    if not cell.is_available:
                        cell_fill_color = "black"

                    rec.set(color=cell_fill_color)

        (reward_ax, policy_ax, observed_ax, utility_ax) = all_axes
        (reward_recs, policy_recs, observed_recs, utility_recs) = all_rectangles

        visualize_rewards(self.world, reward_recs, reward_ax)
        visualize_policy(self.world, self.policy, policy_recs, policy_ax)
        visualize_utilities(self.world, self.utilities, utility_recs, utility_ax)

        figure.canvas.draw()
        figure.canvas.flush_events()

    """
    def policy_evaluation(self: PiterAgentType) -> Dict[Cell, float]:
        # solve this using np
        b: np.ndarray = np.zeros(self.num_available_states, dtype=float)
        A: np.ndarray = np.zeros((self.num_available_states, self.num_available_states), dtype=float)

        for cell, idx in self.state_to_idx_map.items():
            b[idx] = self.world.reward_function.get_reward(cell) # the utilty of the state

            if not cell.is_terminal_state:
                recommended_action = self.policy.recommend(cell)
                for c, prob in TransitionModel.get_transition_probs(self.world, cell, recommended_action).items():
                    A[idx, self.state_to_idx_map[c]] += prob
                # now subtract away idx from whatever coefficient
                A[idx, idx] -= 1
            else:
                A[idx, idx] = 1

        # now solve
        print(A)
        print(b)
        utilities: np.ndarray = np.linalg.solve(A, b)
        print(utilities.shape)

        return {cell: utilities[idx] for cell, idx in self.state_to_idx_map.items()}
    """

    def modified_policy_evaluation(self: PiterAgentType) -> Dict[Cell, float]:
        u: Dict[Cell, float] = cp.copy(self.utilities)
        u_prime: Dict[Cell, float] = cp.copy(self.utilities)

        for _ in range(self.k):
            u = cp.copy(u_prime)

            for cell in u.keys():
                utility = self.world.reward_function.get_reward(cell)

                if not cell.is_terminal_state:
                    recommended_action = self.policy.recommend(cell)

                    for c, prob in TransitionModel.get_transition_probs(self.world, cell, recommended_action).items():
                        utility += self.world.gamma * prob * u[c]

                u_prime[cell] = utility
        return u_prime

    def solve(self: PiterAgentType,
              figure: plt.Figure,
              all_rectangles: List[List[List[plt.Rectangle]]],
              all_axes: List[plt.Axes]
              ) -> None:
        # implement policy iteration

        unchanged: bool = False
        while not unchanged:
            self.utilities = self.modified_policy_evaluation()
            old_policy: Policy = cp.copy(self.policy)
            self.policy.recalculate(self.utilities)

            unchanged = False
            for cell in self.utilities.keys():
                if not cell.is_terminal_state:
                    if old_policy.recommend(cell) != self.policy.recommend(cell):
                        unchanged = True

            self.visualize(figure, all_rectangles, all_axes)
            time.sleep(self.refresh_period)

        self.policy.recalculate(self.utilities)
        self.visualize(figure, all_rectangles, all_axes)

    def play_fit(self: PiterAgentType,
                 figure: plt.Figure,
                 all_rectangles: List[List[List[plt.Rectangle]]],
                 all_axes: List[plt.Axes]
                 ) -> None:
        return


class DirectUtilityEstimationAgent(object):
    def __init__(self: DUEAgentType,
                 world: Grid,
                 refresh_period: float,
                 num_games: int = 4,
                 epsilon: float = 1e-6,
                 max_trajectory_size: int = int(1e6)
                 ) -> None:
        self.world = world
        self.refresh_period = refresh_period
        self.num_games = num_games
        self.epsilon = epsilon
        self.max_trajectory_size = max_trajectory_size

        self.policy = Policy(self.world)
        self.utility_sums: Dict[Cell, float] = {cell: 0.0 for cell_row in self.world.grid
                                                          for cell in cell_row
                                                          if cell.is_available}
        self.num_samples: Dict[Cell, int] = {cell: 0 for cell_row in self.world.grid
                                                     for cell in cell_row
                                                     if cell.is_available}
        self.utilities: Dict[Cell, float] = {cell: 0.0 for cell_row in self.world.grid
                                                       for cell in cell_row
                                                       if cell.is_available}
        self.policy.recalculate(self.utilities)

        self.num_trajectories: int = 0

    def visualize(self: DUEAgentType,
                  figure: plt.Figure,
                  all_rectangles: List[List[List[plt.Rectangle]]],
                  all_axes: List[plt.Axes],
                  **kwargs
                  ) -> None:

        for rectangles in all_rectangles:
            for cell_row, rec_row in zip(self.world.grid, rectangles):
                for cell, rec in zip(cell_row, rec_row):
                    cell_fill_color = "white"
                    if not cell.is_available:
                        cell_fill_color = "black"

                    rec.set(color=cell_fill_color)

        (reward_ax, policy_ax, observed_ax, utility_ax) = all_axes
        (reward_recs, policy_recs, observed_recs, utility_recs) = all_rectangles

        visualize_rewards(self.world, reward_recs, reward_ax)
        visualize_policy(self.world, self.policy, policy_recs, policy_ax)
        visualize_utilities(self.world, self.utilities, utility_recs, utility_ax)
        visualize_frequency_heatmap(self.world, self.num_samples, self.num_trajectories, observed_recs, observed_ax)

        figure.canvas.draw()
        figure.canvas.flush_events()

    def solve(self: DUEAgentType,
              figure: plt.Figure,
              all_rectangles: List[List[List[plt.Rectangle]]],
              all_axes: List[plt.Axes]
              ) -> None:
        return

    def play_game(self: DUEAgentType) -> List[Cell]:
        trajectory: List[Cell] = list()

        self.world.reset()
        cell = self.world.current_cell
        trajectory.append(cell)

        is_over: bool = False
        idx = 0
        while not is_over and idx < self.max_trajectory_size:
            # print(idx, cell, cell.is_terminal_state)
            action = self.policy.recommend(cell)
            cell = self.world.apply_stochastic_action(cell, action)

            trajectory.append(cell)
            is_over = cell.is_terminal_state
            idx += 1

        return trajectory

    def play_fit(self: DUEAgentType,
                 figure: plt.Figure,
                 all_rectangles: List[List[List[plt.Rectangle]]],
                 all_axes: List[plt.Axes]
                 ) -> None:

        is_converged: bool = False
        while not is_converged:

            # play a bunch of games
            trajectories: List[List[Cell]] = list()
            for _ in tqdm(range(self.num_games), total=self.num_games,
                          desc=f"calculating U^(pi)(s) from {self.num_games} games"):
                trajectories.append(self.play_game())

            # now collect sample trajectory rewards
            for trajectory in trajectories:
                for idx, cell in enumerate(trajectory):
                    # expected discounted reward to go
                    utility_to_go: float = self.world.reward_function.get_reward(cell)
                    for t, c in enumerate(trajectory[idx+1:]):
                        utility_to_go += (self.world.gamma ** t) * self.world.reward_function.get_reward(c)

                    self.utility_sums[cell] += utility_to_go
                    self.num_samples[cell] += 1
                self.num_trajectories += 1

            utilities = {cell: self.utility_sums[cell] / self.num_samples[cell]
                         if self.num_samples[cell] > 0 else 0
                         for cell in self.utility_sums.keys()}

            is_converged = True
            for cell in utilities.keys():
                if abs(utilities[cell] - self.utilities[cell]) > self.epsilon:
                    is_converged = False

            self.utilities = utilities
            self.policy.recalculate(self.utilities)
            self.visualize(figure, all_rectangles, all_axes)
            time.sleep(self.refresh_period)


class PassiveAdaptiveDynamicProgrammingAgent(object):
    def __init__(self: PADPAgentType,
                 world: Grid,
                 refresh_period: float,
                 epsilon: float = 1e-6,
                 num_games: int = 4,
                 max_trajectory_size: int = int(1e6),
                 k: int = 15
                 ) -> None:
        self.world = world
        self.refresh_period = refresh_period
        self.epsilon = epsilon
        self.num_games = num_games
        self.max_trajectory_size = max_trajectory_size
        self.k = k

        self.policy = Policy(self.world)

        unique_cells = {c for row in self.world.grid for c in row if c.is_available}

        self.transition_numerators: Dict[Tuple[Cell, Action, Cell], int] = {
            (cell1, a, cell2): 0
            for cell1, cell2 in itertools.product(unique_cells, unique_cells) for a in Action
        }
        self.transition_denominators: Dict[Tuple[Cell, Action], int] = {
            (cell, a): 0
            for cell in unique_cells for a in Action
        }

        self.transition_function: Dict[Tuple[Cell, Action], Dict[Cell, float]] = defaultdict(lambda: defaultdict(float))


        self.num_samples: Dict[Cell, int] = {cell: 0 for cell_row in self.world.grid
                                                     for cell in cell_row
                                                     if cell.is_available}

        self.utilities: Dict[Cell, float] = defaultdict(float)
        self.policy.recalculate(self.utilities)

        self.num_trajectories: int = 0


    def visualize(self: PADPAgentType,
                  figure: plt.Figure,
                  all_rectangles: List[List[List[plt.Rectangle]]],
                  all_axes: List[plt.Axes],
                  utilities: Dict[Cell, float] = None,
                  **kwargs
                  ) -> None:

        if utilities is None:
            utilities = self.utilities

        for rectangles in all_rectangles:
            for cell_row, rec_row in zip(self.world.grid, rectangles):
                for cell, rec in zip(cell_row, rec_row):
                    cell_fill_color = "white"
                    if not cell.is_available:
                        cell_fill_color = "black"

                    rec.set(color=cell_fill_color)

        (reward_ax, policy_ax, observed_ax, utility_ax) = all_axes
        (reward_recs, policy_recs, observed_recs, utility_recs) = all_rectangles

        visualize_rewards(self.world, reward_recs, reward_ax)
        visualize_policy(self.world, self.policy, policy_recs, policy_ax)
        visualize_utilities(self.world, utilities, utility_recs, utility_ax)
        visualize_frequency_heatmap(self.world, self.num_samples, self.num_trajectories, observed_recs, observed_ax)

        figure.canvas.draw()
        figure.canvas.flush_events()

    def solve(self: PADPAgentType,
              figure: plt.Figure,
              all_rectangles: List[List[List[plt.Rectangle]]],
              all_axes: List[plt.Axes]
              ) -> None:
        return

    def modified_policy_evaluation(self: PiterAgentType,
                                   utilities: Dict[Cell, float]
                                   ) -> Dict[Cell, float]:
        u: Dict[Cell, float] = cp.copy(utilities)
        u_prime: Dict[Cell, float] = cp.copy(utilities)

        for _ in range(self.k):
            u = cp.copy(u_prime)

            for cell in u.keys():
                utility = self.world.reward_function.get_reward(cell)

                if not cell.is_terminal_state:
                    recommended_action = self.policy.recommend(cell)

                    for c, prob in self.transition_function[(cell, recommended_action)].items():
                        utility += self.world.gamma * prob * u[c]

                u_prime[cell] = utility
        return u_prime

    def play_game(self: PADPAgentType,
                  figure: plt.Figure,
                  all_rectangles: List[List[List[plt.Rectangle]]],
                  all_axes: List[plt.Axes]
                  ) -> Dict[Cell, float]:
        utilities: Dict[Cell, float] = self.utilities

        self.world.reset()
        cell = self.world.current_cell

        if cell not in utilities:
            utilities[cell] = self.world.reward_function.get_reward(cell)
        utilities = self.modified_policy_evaluation(utilities)

        self.visualize(figure, all_rectangles, all_axes, utilities=utilities)
        time.sleep(self.refresh_period)

        is_over: bool = False
        idx = 0
        while not is_over and idx < self.max_trajectory_size:
            self.num_samples[cell] += 1

            # print(idx, cell, cell.is_terminal_state)
            action = self.policy.recommend(cell)
            next_cell = self.world.apply_stochastic_action(cell, action)
            reward = self.world.reward_function.get_reward(next_cell)

            if next_cell not in utilities:
                utilities[next_cell] = self.world.reward_function.get_reward(next_cell)
            self.transition_denominators[(cell, action)] += 1
            self.transition_numerators[(cell, action, next_cell)] += 1

            # update transition function
            for t in utilities.keys():
                if self.transition_numerators[(cell, action, t)] > 0:
                    n = self.transition_numerators[(cell, action, t)]
                    d = self.transition_denominators[(cell, action)]
                    self.transition_function[(cell, action)][t] = n/d

            utilities = self.modified_policy_evaluation(utilities)

            self.visualize(figure, all_rectangles, all_axes, utilities=utilities)
            time.sleep(self.refresh_period)

            # print(cell)
            cell = next_cell
            is_over = cell.is_terminal_state
            idx += 1

        if is_over:
            self.num_samples[cell] += 1
            self.visualize(figure, all_rectangles, all_axes, utilities=utilities)
            time.sleep(self.refresh_period)

        self.num_trajectories += 1

        return utilities

    def play_fit(self: PADPAgentType,
                 figure: plt.Figure,
                 all_rectangles: List[List[List[plt.Rectangle]]],
                 all_axes: List[plt.Axes]
                 ) -> None:

        is_converged: bool = False
        while not is_converged:

            # play a bunch of games
            for _ in tqdm(range(self.num_games), total=self.num_games,
                          desc=f"calculating U^(pi)(s) from {self.num_games} games"):
                utilities = self.play_game(figure, all_rectangles, all_axes)

            self.policy.recalculate(utilities)

            is_converged = True
            for cell in utilities.keys():
                if abs(utilities[cell] - self.utilities[cell]) > self.epsilon:
                    is_converged = False

            self.utilities = utilities
            self.policy.recalculate(self.utilities)
            self.visualize(figure, all_rectangles, all_axes)
            time.sleep(self.refresh_period)


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("num_x_coords", type=int, help="number of x coords in maze")
    parser.add_argument("num_y_coords", type=int, help="number of y coords in maze")

    parser.add_argument("agent", type=str, default="viter", choices=["viter", "piter", "due", "padp"])

    parser.add_argument("terminal_coords", nargs="+", type=int, help="coordinates of terminal states")
    parser.add_argument("--terminal_rewards", nargs="+", type=float, default=None, help="rewards for terminal states")

    parser.add_argument("--start_x", type=int, default=0, help="x coord of start state")
    parser.add_argument("--start_y", type=int, default=0, help="y coord of start state")

    parser.add_argument("--nonterm_reward", type=float, default=-0.04, help="nonterminal reward")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument("--obstacle_coords", nargs="+", type=int, help="coordinates of cells that are unavailable")
    parser.add_argument("--refresh_period", type=float, default=0.5, help="num secs to wait before updating")
    parser.add_argument("--num_games", type=int, default=1, help="number of games to play at once")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="convergence threshold")

    args = parser.parse_args()

    CELL_WIDTH: int = 1

    world: Grid = Grid(args.num_x_coords,
                      args.num_y_coords,
                      init_x=args.start_x,
                      init_y=args.start_y,
                      obstacle_coords_list=args.obstacle_coords,
                      terminal_coords_list=args.terminal_coords,
                      terminal_rewards_list=args.terminal_rewards,
                      nonterm_reward=args.nonterm_reward,
                      gamma=args.gamma)
    agent: object = None

    if args.agent == "viter":
        agent = ValueIterationAgent(world, args.refresh_period, epsilon=args.epsilon)
    elif args.agent == "piter":
        agent = PolicyIterationAgent(world, args.refresh_period)
    elif args.agent == "due":
        agent = DirectUtilityEstimationAgent(world, args.refresh_period, num_games=args.num_games, epsilon=args.epsilon)
    elif args.agent == "padp":
        agent = PassiveAdaptiveDynamicProgrammingAgent(world,
                                                       args.refresh_period,
                                                       num_games=args.num_games,
                                                       epsilon=args.epsilon)
    else:
        raise Exception("ERROR: unknown agent type [{0}]".format(args.agent))

    # pprint(args.terminal_coords)
    # pprint(TransitionModel.get_transition_probs(grid, grid.get_cell(0, 0), Action.UP))
    # pprint(TransitionModel.get_transition_probs(grid, grid.get_cell(1, 0), Action.UP))
    # pprint(grid.terminal_cells)

    # agent.solve()
    # pprint(agent.policy.policy_map)
    # return

    plt.ion()
    figure, ((reward_ax, policy_ax), (observed_ax, utility_ax)) = plt.subplots(2, 2)

    reward_ax.set_title("rewards")
    policy_ax.set_title("policy")
    observed_ax.set_title("observed occupancy heatmap")
    utility_ax.set_title("utility heatmap")

    all_axes = (reward_ax, policy_ax, observed_ax, utility_ax)
    all_rectangles = make_rectangles(world, all_axes)

    for ax in all_axes:
        # ax.set_xlim(left=0, right=grid.num_x_coords)
        # ax.set_ylim(bottom=0, top=grid.num_y_coords)
        ax.set_axis_off()

    # ax.set_xticklabels([i for i in range(grid.num_x_coords)])

    for ax, rectangles in zip(all_axes, all_rectangles):
        for row in rectangles:
            for rec in row:
                ax.add_artist(rec)

    agent.visualize(figure, all_rectangles, all_axes)
    visualize_boundaries(world, all_axes)
    figure.canvas.draw()
    figure.canvas.flush_events()
    # time.sleep(0.1)


    # time.sleep(5)
    agent.solve(figure, all_rectangles, all_axes)

    # time.sleep(5)
    agent.play_fit(figure, all_rectangles, all_axes)

    while True:
        # print("hi")
        # agent.visualize(figure, all_rectangles, all_axes)
        figure.canvas.draw()
        figure.canvas.flush_events()

    return


if __name__ == "__main__":
    main()

