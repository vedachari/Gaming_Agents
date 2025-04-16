# SYSTEM IMPORTS
from enum import Enum, unique
from collections import defaultdict
from typing import Dict, Set, Type, Tuple, Union, List
import numpy as np


# PYTHON PROJECT IMPORTS


ActionType = Type["Action"]
CellType = Type["Cell"]
TransType = Type["TransitionModel"]
RewardType = Type["RewardFunction"]
GridType = Type["Grid"]

@unique
class Action(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    @staticmethod
    def invert(a: ActionType) -> ActionType:
        a_inverse: Action = None

        if a == Action.UP:
            a_inverse = Action.DOWN
        elif a == Action.DOWN:
            a_inverse = Action.UP
        elif a == Action.LEFT:
            a_inverse = Action.RIGHT
        elif a == Action.RIGHT:
            a_inverse = Action.LEFT
        else:
            raise Exception("ERROR: unknown action [{0}]".format(a))

        return a_inverse

    @staticmethod
    def is_perpendicular(a: ActionType,
                         b: ActionType
                         ) -> bool:
        is_perp: bool = False
        if a == Action.UP or a == Action.DOWN:
            is_perp = b == Action.LEFT or b == Action.RIGHT
        elif a == Action.LEFT or a == Action.RIGHT:
            is_perp = b == Action.UP or b == Action.DOWN
        return is_perp

    @staticmethod
    def apply(cell: CellType,
              action: ActionType
              ) -> CellType:
        next_cell: Cell = None
        if action == Action.UP:
            next_cell = Cell(cell.x, cell.y+1)
        elif action == Action.DOWN:
            next_cell = Cell(cell.x, cell.y-1)
        elif action == Action.LEFT:
            next_cell = Cell(cell.x-1, cell.y)
        elif action == Action.RIGHT:
            next_cell = Cell(cell.x+1, cell.y)
        else:
            raise Exception("ERROR: unknown action [{0}]".format(action))
        return next_cell


# code for the Cell and Maze classes was adapted from https://scipython.com/blog/making-a-maze/
class Cell(object):
    def __init__(self: CellType,
                 x: int,
                 y: int
                 ) -> None:
        self.x: int = x
        self.y: int = y

        # if the square is an obstacle (i.e. not available) then this will be False
        self.is_available: bool = True
        self.is_start_state: bool = False
        self.is_terminal_state: bool = False

    def __hash__(self: CellType) -> int:
        return (self.x, self.y).__hash__()

    def __eq__(self: CellType,
               other: CellType) -> bool:
        return (self.x, self.y) == (other.x, other.y)

    def __str__(self: CellType) -> str:
        return "({0}, {1})".format(self.x, self.y)

    def __repr__(self: CellType) -> str:
        return str(self)


class TransitionModel(object):

    @staticmethod
    def get_transition_probs(g: GridType,
                             c: Cell,
                             a: Action
                             ) -> Dict[Cell, float]:
        d: Dict[Cell, float] = defaultdict(float)
        for b in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            if a == b:
                d[g.apply_deterministic_action(c, b)] += 0.8
            elif Action.is_perpendicular(a, b):
                d[g.apply_deterministic_action(c, b)] += 0.1
        return d

class RewardFunction(object):
    def __init__(self: RewardType,
                 terminal_cells: List[CellType],
                 terminal_rewards: List[float],
                 nonterm_reward: float
                 ) -> None:
        self.nonterm_reward: float = nonterm_reward
        self.cell_to_rewards = {c: r for c,r in zip(terminal_cells, terminal_rewards)}

    def get_reward(self: RewardType,
                   c: CellType
                   ) -> float:
        return self.cell_to_rewards.get(c, self.nonterm_reward)


class Grid(object):
    def __init__(self: GridType,
                 num_x_coords: int,
                 num_y_coords: int,
                 init_x: int = 0,
                 init_y: int = 0,
                 nonterm_reward: float = -0.04,
                 gamma: float = 0.99,
                 terminal_coords_list: List[int] = None,
                 terminal_rewards_list: List[int] = None,
                 obstacle_coords_list: List[int] = None
                 ) -> None:
        self.num_x_coords: int = num_x_coords
        self.num_y_coords: int = num_y_coords

        # start state
        self.init_x: int = init_x
        self.init_y: int = init_y

        # discount
        self.gamma = gamma

        # the grid
        self.grid: List[List[Cell]] = [[Cell(x, y) for y in range(self.num_y_coords)]
                                        for x in range(self.num_x_coords)]

        # current agent state
        self.current_cell: Cell = self.get_cell(self.init_x, self.init_y)
        self.current_cell.is_start_state = True

        # set the terminal states
        self.terminal_coords_list = terminal_coords_list
        self.terminal_cells: List[Cell] = list()
        # print(self.terminal_coords_list)
        if self.terminal_coords_list is not None:
            assert(len(self.terminal_coords_list) % 2 == 0)
            for idx in range(int(len(self.terminal_coords_list) / 2)):
                x_coord = self.terminal_coords_list[2*idx]
                y_coord = self.terminal_coords_list[2*idx+1]

                # print(x_coord, y_coord)

                assert(self.get_cell(x_coord, y_coord).is_start_state == False)
                self.get_cell(x_coord, y_coord).is_terminal_state = True
                self.terminal_cells.append(self.get_cell(x_coord, y_coord))

        self.terminal_rewards_list = terminal_rewards_list
        if self.terminal_rewards_list is not None:
            assert(len(self.terminal_rewards_list) == len(self.terminal_cells))
        else:
            self.terminal_rewards_list = [1 for _ in self.terminal_cells]
        self.reward_function = RewardFunction(self.terminal_cells, self.terminal_rewards_list, nonterm_reward)

        # apply obstacles if they exist
        self.obstacle_coords_list = obstacle_coords_list
        if self.obstacle_coords_list is not None:
            # print(self.obstacle_coords_list)
            assert(len(self.obstacle_coords_list) % 2 == 0)
            for idx in range(int(len(self.obstacle_coords_list) / 2)):
                x_coord = self.obstacle_coords_list[2*idx]
                y_coord = self.obstacle_coords_list[2*idx+1]

                # assert obstacle cannot be a start or goal state
                assert(self.get_cell(x_coord, y_coord).is_start_state == False)
                assert(self.get_cell(x_coord, y_coord).is_terminal_state == False)
                self.get_cell(x_coord, y_coord).is_available = False

    def reset(self: GridType) -> None:
        self.current_cell = self.get_cell(self.init_x, self.init_y)

    def is_on_edge(self: GridType,
                   cell: Cell
                   ) -> bool:
        return cell.x == 0 or cell.x == self.num_x_coords-1 or cell.y == 0 or cell.y == self.num_x_coords-1

    def get_cell(self: GridType,
                 x: int,
                 y: int
                 ) -> Cell:
        return self.grid[x][y]

    def apply_deterministic_action(self: GridType,
                                   cell: Cell,
                                   action: Action
                                   ) -> Cell:

        new_cell = Action.apply(cell, action)

        if new_cell.x < 0 or new_cell.x >= self.num_x_coords\
           or new_cell.y < 0 or new_cell.y >= self.num_y_coords or\
           not self.get_cell(new_cell.x, new_cell.y).is_available:
            new_cell = cell
        else:
            new_cell = self.get_cell(new_cell.x, new_cell.y)
        return new_cell

    def apply_stochastic_action(self: GridType,
                                cell: Cell,
                                action: Action
                                ) -> Cell:
        cells: List[Cell] = list()
        probs: List[float] = list()
        for c, prob in TransitionModel.get_transition_probs(self, cell, action).items():
            cells.append(c)
            probs.append(prob)

        return np.random.choice(cells, p=probs)

