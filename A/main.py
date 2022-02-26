import sys
import argparse
import time
import random
import logging
from typing import Tuple, List
from collections import deque

input = sys.stdin.readline
logger = logging.getLogger(__name__)
OP2DIST = {
    "u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1),  # 仕切りを設置
    "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1),  # 移動
    ".": (0, 0)                                            # 何もしない
}
DIST2PUT = {
    (-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"
}
UDLR_MOVE = "UDLR"
UDLR_PUT = "udlr"
REVERSE_OP = {
    "u": "d", "d": "u", "l": "r", "r": "l",  # 仕切りを設置
    "U": "D", "D": "U", "L": "R", "R": "L",  # 移動
    ".": "."                                 # 何もしない
}

SPACE = 0
FENCE = 1

MAX_X = 32
MAX_Y = 32


class Pet():

    COW = 1
    PIG = 2
    RABBIT = 3
    DOG = 4
    CAT = 5
    PET_TYPE = ["NONE", "COW", "PIG", "RABBIT", "DOG", "CAT"]

    def __init__(self, px: int, py: int, pt: int) -> None:
        self.px = px
        self.py = py
        self.pt = pt
    
    def update(self, ops: List[str]) -> None:
        """ターン終了時に呼ばれる。座標を更新する。"""
        px = self.px
        py = self.py
        for op in ops:
            if op == "U":
                px -= 1
            if op == "D":
                px += 1
            if op == "L":
                py -= 1
            if op == "R":
                py += 1
        self.px = px
        self.py = py
    
    def __str__(self) -> str:
        return f"{self.PET_TYPE[self.pt]}({self.px}, {self.py})"
    
    def __repr__(self) -> str:
        return str(self)


class Human():

    def __init__(self, hx: int, hy: int) -> None:
        self.hx = hx
        self.hy = hy
        self.next_op = "x"
        self.next_hx = hx
        self.next_hy = hy
    
    def set_op(self, room, op: str) -> None:
        if op in UDLR_PUT:
            dx, dy = OP2DIST[op]
            fx = self.hx + dx
            fy = self.hy + dy
            room[fx][fy] = FENCE
        self.next_op = op

        hx = self.hx
        hy = self.hy
        if self.next_op == "U":
            hx -= 1
        if self.next_op == "D":
            hx += 1
        if self.next_op == "L":
            hy -= 1
        if self.next_op == "R":
            hy += 1
        self.next_hx = hx
        self.next_hy = hy
    
    def update(self):
        """ターン終了時に呼ばれる。座標を更新する。"""
        self.hx = self.next_hx
        self.hy = self.next_hy
        self.next_op = "x"
    
    def __str__(self) -> str:
        return f"({self.hx}, {self.hy}, {self.next_op})"
    
    def __repr__(self) -> str:
        return str(self)


class Strategy():

    def __init__(self, room, N: int, pets: List[Pet], M: int, humen: List[Human]) -> None:
        self.room = room
        self.N = N
        self.pets = pets
        self.M = M
        self.humen = humen
    
    def bfs_all(self, room, sx: int, sy: int) -> List[List[int]]:
        """(sx, sy) からの距離を求める"""
        distance = [[-1 for y in range(MAX_Y)] for x in range(MAX_X)]
        q = deque()
        q.append((sx, sy))
        distance[sx][sy] = 0
        while q:
            x, y = q.popleft()
            for op_move in UDLR_MOVE:
                dx, dy = OP2DIST[op_move]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < MAX_X and 0 <= ny < MAX_Y):
                    continue
                elif distance[nx][ny] != -1:
                    continue
                elif room[nx][ny] == SPACE:
                    distance[nx][ny] = distance[x][y] + 1
                    q.append((nx, ny))
        return distance
    
    def is_reachable(self, distance: List[List[int]], tx: int, ty: int) -> bool:
        """(tx, ty) へ移動可能ならTrue"""
        return distance[tx][ty] != -1
    
    def get_ops_moveto(self, distance: List[List[int]], tx: int, ty: int) -> List[str]:
        """(tx, ty)への移動手順を求める"""
        x = tx
        y = ty
        if not self.is_reachable(distance, tx, ty):
            return ["."]
        ops = []
        while True:
            if distance[x][y] == 0:
                break
            for op_move in UDLR_MOVE:
                dx, dy = OP2DIST[op_move]
                nx, ny = x - dx, y - dy
                if not (0 <= nx < MAX_X and 0 <= ny < MAX_Y):
                    continue
                if distance[nx][ny] == distance[x][y] - 1:
                    ops.append(op_move)
                    x, y = nx, ny
                    break
        return ops[::-1]
    
    def get_ops_moveneighbor(self, distance: List[List[int]], tx: int, ty: int) -> List[str]:
        """(tx, ty)に柵を置くため、手前のマスへの移動手順を求める"""
        ops = self.get_ops_moveto(distance, tx, ty)
        if distance[tx][ty] >= 1:
            return ops[:-1]
        else:
            for op_move in UDLR_MOVE:
                dx, dy = OP2DIST[op_move]
                nx, ny = tx + dx, ty + dy
                if not (0 <= nx < MAX_X and 0 <= ny < MAX_Y):
                    continue
                elif distance[nx][ny] == 1:
                    return op_move
            return "."
    
    def is_puttable(self, room, hi: int, x: int, y: int, allow_separation=False) -> bool:
        """人がこのターンに座標(x, y)を通行止めにできるときTrue"""
        if self.room[x][y] == FENCE:
            return False
        
        for human in self.humen:
            if x == human.hx and y == human.hy:
                return False
            if x == human.next_hx and y == human.next_hy:
                return False
            
        for pet in self.pets:
            px = pet.px
            py = pet.py
            if x == px and y == py:
                return False
            if x == px+1 and y == py:
                return False
            if x == px and y == py+1:
                return False
            if x == px-1 and y == py:
                return False
            if x == px and y == py-1:
                return False
        
        if not allow_separation:
            fenced_room = [[room[x][y] for y in range(MAX_Y)] for x in range(MAX_X)]
            fenced_room[x][y] = FENCE
            hx = self.humen[hi].hx
            hy = self.humen[hi].hy
            fenced_distance = self.bfs_all(fenced_room, hx, hy)
            # 通行止め後、人が別々の領域に分かれてはならない
            for gi in range(self.M):
                gx = self.humen[gi].hx
                gy = self.humen[gi].hy
                if not self.is_reachable(fenced_distance, gx, gy):
                    return False
        return True

    def update(self) -> None:
        """ターン終了時に呼ばれる。戦略を更新する。"""
        pass

    def get_op(self, hi: int) -> str:
        """次の1手を選ぶ。"""
        return "."

    def is_terminate(self) -> bool:
        """戦略の終了を通知するときTrue"""
        return False

class BuildTrapStrategy(Strategy):
    """犬を閉じ込める罠を1人1つ作成する"""

    N_LANES = 3  # N_LANES <= 5
    LANE_Y_MIN = 5
    LANE_Y_MAX = MAX_Y-5

    def __init__(self, room, N: int, pets: List[Pet], M: int, humen: List[Human]) -> None:
        super().__init__(room, N, pets, M, humen)
        self.targets = [deque() for i in range(self.M)]
        for hi in range(self.N_LANES):
            x = hi * 2 + 2
            if hi % 2 == 0:
                for y in range(self.LANE_Y_MIN, self.LANE_Y_MAX, 1):
                    self.targets[hi].append((x, y))
            else:
                for y in range(self.LANE_Y_MAX-1, self.LANE_Y_MIN-1, -1):
                    self.targets[hi].append((x, y))
        self.n_dogs = sum(pet.pt == Pet.DOG for pet in self.pets)
    
    def is_terminate(self) -> bool:
        """戦略行動が完了したらTrue"""
        return self.n_dogs == 0 or all(len(self.targets[hi]) == 0 for hi in range(self.M))

    def update(self) -> None:
        """ターン終了時に呼ばれる。戦略を更新する。"""
        for hi in range(self.M):
            if len(self.targets[hi]) > 0:
                tx, ty = self.targets[hi][0]
                if self.room[tx][ty] == FENCE:
                    self.targets[hi].popleft()
            
    def get_op(self, hi: int) -> str:
        """次の1手を選ぶ。targetに移動して柵を設置する。柵を設置し終わったら停止する。"""
        hx = self.humen[hi].hx
        hy = self.humen[hi].hy
        op = "."
        distance = self.bfs_all(self.room, hx, hy)

        if len(self.targets[hi]) == 0:
            if hi % 2 == 0:
                wx, wy = (1, self.LANE_Y_MAX)
            else:
                wx, wy = (1, self.LANE_Y_MIN - 1)
            ops = self.get_ops_moveto(distance, wx, wy)
            op = ops[0] if len(ops) else "."
            return op
        
        tx, ty = self.targets[hi][0]
        dx = tx - hx
        dy = ty - hy
        if abs(dx) + abs(dy) == 1:
            if self.is_puttable(self.room, hi, tx, ty):
                op = DIST2PUT[(dx, dy)]
        else:
            ops = self.get_ops_moveneighbor(distance, tx, ty)
            op = ops[0] if len(ops) else "."
        return op

class ActivateTrapStrategy(Strategy):
    """罠を使う"""

    N_LANES = 3  # N_LANES <= 5
    LANE_Y_MIN = 5
    LANE_Y_MAX = MAX_Y-5

    def __init__(self, room, N: int, pets: List[Pet], M: int, humen: List[Human]) -> None:
        super().__init__(room, N, pets, M, humen)
        self.waypoints = [deque() for i in range(self.M)]
        self.targets = [deque() for i in range(self.M)]
        for hi in range(self.M):
            for i in range(self.N_LANES):
                x = i * 2 + 1
                if hi % 2 == 0:
                    self.waypoints[hi].append((x, self.LANE_Y_MAX))
                    self.targets[hi].append((x, self.LANE_Y_MAX - 1))
                else:
                    self.waypoints[hi].append((x, self.LANE_Y_MIN - 1))
                    self.targets[hi].append((x, self.LANE_Y_MIN))
        self.n_dogs = sum(pet.pt == Pet.DOG for pet in self.pets)
        self.n_separated_dogs = self.get_n_separated_pets()[Pet.DOG]
        self.n_lanes_left = self.N_LANES
        self.is_put = False
    
    def is_terminate(self) -> bool:
        """戦略行動が完了したらTrue"""
        return self.n_dogs - self.n_separated_dogs == 0 or self.n_lanes_left == 0
    
    def is_gathered(self) -> bool:
        """全員が目的地に到着しているときTrue"""
        if self.n_lanes_left == 0:
            return True
        return all([(self.humen[hi].hx, self.humen[hi].hy) == self.waypoints[hi][0] for hi in range(self.M)])
    
    def get_n_falled_into_trap(self) -> List[int]:
        """罠にかかった動物を数える"""
        n_falled = [0] * 6
        x = (self.N_LANES - self.n_lanes_left) * 2 + 1
        for pet in self.pets:
            if pet.px == x and (self.LANE_Y_MIN + 2 <= pet.py < self.LANE_Y_MAX - 2):
                n_falled[pet.pt] += 1
        return n_falled
    
    def get_n_separated_pets(self) -> List[int]:
        """隔離済みの動物を数える"""
        n_separated = [0] * 6
        hx = self.humen[0].hx
        hy = self.humen[0].hy
        distance = self.bfs_all(self.room, hx, hy)
        for pet in self.pets:
            if not self.is_reachable(distance, pet.px, pet.py):
                n_separated[pet.pt] += 1
        return n_separated
    
    def update(self) -> None:
        """ターン終了時に呼ばれる。戦略を更新する。"""
        for hi in range(self.M):
            if len(self.targets[hi]) > 0:
                tx, ty = self.targets[hi][0]
                if self.room[tx][ty] == FENCE:
                    self.waypoints[hi].popleft()
                    self.targets[hi].popleft()
        if self.is_put:
            n_separated = self.get_n_separated_pets()
            self.n_separated_dogs = n_separated[Pet.DOG]
            self.n_lanes_left -= 1
        self.is_put = False

    def get_op(self, hi: int) -> str:
        """次の1手を選ぶ。targetに移動して、犬を検知したら柵を設置する。"""
        if len(self.targets[hi]) == 0:
            return "."
        hx = self.humen[hi].hx
        hy = self.humen[hi].hy
        op = "."
        distance = self.bfs_all(self.room, hx, hy)
        wx, wy = self.waypoints[hi][0]
        if hx == wx and hy == wy:
            # 2人で協調して柵を設置する
            if self.is_put:
                if hi == 0:
                    op = "l"
                elif hi == 1:
                    op = "r"
                else:
                    op = "."
            elif self.is_gathered() and any(self.get_n_falled_into_trap()):
                gi = hi - 1 if hi > 0  else 1 - hi
                tx, ty = self.targets[hi][0]
                sx, sy = self.targets[gi][0]
                if self.is_puttable(self.room, hi, tx, ty) and self.is_puttable(self.room, gi, sx, sy):
                    self.is_put = True
                    if hi == 0:
                        op = "l"
                    elif hi == 1:
                        op = "r"
                    else:
                        op = "."
                else:
                    op = "."
        else:
            ops = self.get_ops_moveto(distance, wx, wy)
            op = ops[0] if len(ops) else "."
        return op

class CrossStrategy(Strategy):
    """柵を十字に張り巡らす"""
    def __init__(self, room, N: int, pets: List[Pet], M: int, humen: List[Human]) -> None:
        super().__init__(room, N, pets, M, humen)
        self.targets = [deque() for i in range(self.M)]
        for hi in range(self.M):
            hx = self.humen[hi].hx
            hy = self.humen[hi].hy
            distance = self.bfs_all(self.room, hx, hy)
            if hi % 4 == 0:
                x = MAX_X // 2
                for y in range(1, MAX_Y//2 - 1):
                    if self.is_reachable(distance, x, y) and distance[x][y] != FENCE:
                        self.targets[hi].append((x, y))
            elif hi % 4 == 1:
                x = MAX_X // 2
                for y in range(MAX_Y//2 + 2, MAX_Y - 1):
                    if self.is_reachable(distance, x, y) and distance[x][y] != FENCE:
                        self.targets[hi].append((x, y))
            elif hi % 4 == 2:
                y = MAX_Y // 2
                for x in range(1, MAX_X//2 - 1):
                    if self.is_reachable(distance, x, y) and distance[x][y] != FENCE:
                        self.targets[hi].append((x, y))
            else:
                y = MAX_Y // 2
                for x in range(MAX_X//2 + 2, MAX_X - 1):
                    if self.is_reachable(distance, x, y) and distance[x][y] != FENCE:
                        self.targets[hi].append((x, y))
    
    def is_terminate(self) -> bool:
        """戦略行動が完了したらTrue"""
        return all([len(self.targets[hi]) == 0 for hi in range(self.M)])

    def update(self) -> None:
        """ターン終了時に呼ばれる。戦略を更新する。"""
        for hi in range(self.M):
            if len(self.targets[hi]) > 0:
                tx, ty = self.targets[hi][0]
                if self.room[tx][ty] == FENCE:
                    self.targets[hi].popleft()
            
    def get_op(self, hi: int) -> str:
        """次の1手を選ぶ。targetに移動して柵を設置する。柵を設置し終わったら停止する。"""
        if len(self.targets[hi]) == 0:
            return "."
        hx = self.humen[hi].hx
        hy = self.humen[hi].hy
        op = "."
        distance = self.bfs_all(self.room, hx, hy)
        tx, ty = self.targets[hi][0]
        dx = tx - hx
        dy = ty - hy
        if abs(dx) + abs(dy) == 1:
            if self.is_puttable(self.room, hi, tx, ty, allow_separation=True):
                op = DIST2PUT[(dx, dy)]
        else:
            ops = self.get_ops_moveneighbor(distance, tx, ty)
            op = ops[0] if len(ops) else "."
        return op

class CloseCrossStrategy(Strategy):
    """十字の柵を閉じる"""
    
    def __init__(self, room, N: int, pets: List[Pet], M: int, humen: List[Human]) -> None:
        super().__init__(room, N, pets, M, humen)
        self.waypoints = [deque() for i in range(self.M)]
        self.targets = [deque() for i in range(self.M)]
        cx = MAX_X // 2
        cy = MAX_Y // 2
        for hi in range(self.M):
            if hi % 4 == 0:
                self.waypoints[hi].append((cx-1, cy))
                self.targets[hi].append((cx-1, cy+1))
            elif hi % 4 == 1:
                self.waypoints[hi].append((cx, cy+1))
                self.targets[hi].append((cx+1, cy+1))
            if hi % 4 == 2:
                self.waypoints[hi].append((cx+1, cy))
                self.targets[hi].append((cx+1, cy-1))
            else:
                self.waypoints[hi].append((cx, cy-1))
                self.targets[hi].append((cx-1, cy-1))
        self.n_fences_left = 3
    
    def get_n_reachable_pets(self, room) -> int:
        """到達可能な動物を数える"""
        n_reacheable = 0
        hx = self.humen[0].hx
        hy = self.humen[0].hy
        distance = self.bfs_all(room, hx, hy)
        for pet in self.pets:
            if self.is_reachable(distance, pet.px, pet.py):
                n_reacheable += 1
        return n_reacheable
    
    def is_puttable(self, room, hi: int, x: int, y: int, allow_separation=False) -> bool:
        if self.n_fences_left <= 0:
            return False
        return super().is_puttable(room, hi, x, y, allow_separation)
    
    def is_terminate(self) -> bool:
        """戦略行動が完了したらTrue"""
        return self.n_fences_left <= 0
    
    def is_gathered(self) -> bool:
        """全員が目的地に到着しているときTrue"""
        if self.n_fences_left <= 0:
            return False
        return all([len(self.waypoints[hi]) == 0 or (self.humen[hi].hx, self.humen[hi].hy) == self.waypoints[hi][0] for hi in range(self.M)])

    def get_n_separatable_pets(self, room) -> int:
        """隔離される予定の動物を数える"""
        n_separatable = 0
        hx = self.humen[0].hx
        hy = self.humen[0].hy
        distance = self.bfs_all(room, hx, hy)
        for pet in self.pets:
            if not self.is_reachable(distance, pet.px, pet.py):
                n_separatable += 1
        return n_separatable
    
    def update(self) -> None:
        for hi in range(self.M):
            if len(self.targets[hi]) > 0:
                tx, ty = self.targets[hi][0]
                if self.room[tx][ty] == FENCE:
                    self.waypoints[hi].popleft()
                    self.targets[hi].popleft()
    
    def get_op(self, hi: int) -> str:
        """次の1手を選ぶ。targetに移動して、犬を検知したら柵を設置する。"""
        if len(self.targets[hi]) == 0:
            return "."
        if self.is_terminate():
            return "."
        hx = self.humen[hi].hx
        hy = self.humen[hi].hy
        op = "."
        if self.is_gathered():
            tx, ty = self.targets[hi][0]
            room_plan = [[self.room[x][y] for y in range(MAX_Y)] for x in range(MAX_X)]
            n_reachable_before = self.get_n_reachable_pets(room_plan)
            room_plan[tx][ty] = FENCE
            n_reachable_after = self.get_n_reachable_pets(room_plan)
            if self.n_fences_left >= 0 and n_reachable_before > 0:
                score = (n_reachable_before - n_reachable_after) / (n_reachable_before / (self.n_fences_left+1))
            else:
                score = 0
            if score > 1.0 and self.is_puttable(self.room, hi, tx, ty):
                dx = tx - hx
                dy = ty - hy
                op = DIST2PUT[(dx, dy)]
                self.n_fences_left -= 1
        else:
            wx, wy = self.waypoints[hi][0]
            distance = self.bfs_all(self.room, hx, hy)
            ops = self.get_ops_moveto(distance, wx, wy)
            op = ops[0] if len(ops) else "."
        return op
    

class ChaseStrategy(Strategy):

    MAX_TURN_CHASING = 10
    RADIUS = 2
    N_PET_MOVES = {Pet.COW: 1, Pet.PIG: 2, Pet.RABBIT: 3, Pet.CAT: 2, Pet.DOG: 2}

    def __init__(self, room, N: int, pets: List[Pet], M: int, humen: List[Human]) -> None:
        super().__init__(room, N, pets, M, humen)
        self.targets = [(-1, -1) for i in range(M)]
        self.turn_chasing = [random.randint(self.MAX_TURN_CHASING // 2, self.MAX_TURN_CHASING) for i in range(self.M)]
        self.pi = -1
        self.switch_pi()
        self.update()

    def switch_pi(self):
        """ペットを切り替える"""
        # TODO: 少ない仕切りで囲えるペットを選ぶ。
        pi = (self.pi + 1) % self.N
        for _ in range(self.N):
            px = self.pets[pi].px
            py = self.pets[pi].py
            pt = self.pets[pi].pt
            if pt == Pet.DOG:
                pi = (pi + 1) % self.N
                continue
            pet_distance = self.bfs_all(self.room, px, py)
            is_separated = all([not self.is_reachable(pet_distance, human.hx, human.hy) for human in self.humen])
            if is_separated:
                pi = (pi + 1) % self.N
                continue
        self.pi = pi

    def update(self) -> None:
        """ターン終了時に呼ばれる。戦略を更新する。"""
        for hi in range(self.M):
            self.turn_chasing[hi] -= 1
        px = self.pets[self.pi].px
        py = self.pets[self.pi].py
        pet_distance = self.bfs_all(self.room, px, py)

        # 閉じ込めに成功したら次のペットに対象を切り替える
        is_separated = all([not self.is_reachable(pet_distance, human.hx, human.hy) for human in self.humen])
        if is_separated:
            self.switch_pi()
            for hi in range(self.M):
                self.targets[hi] = (-1, -1)
                self.turn_chasing[hi] = random.randint(self.MAX_TURN_CHASING // 2, self.MAX_TURN_CHASING)
        
        # 目標にフェンスが置かれている場合、もしくは長時間かけても到達しない場合、目標をリセットする
        for hi in range(self.M):
            tx, ty = self.targets[hi]
            if self.room[tx][ty] == FENCE or self.turn_chasing[hi] <= 0:
                self.targets[hi] = (-1, -1)
                self.turn_chasing[hi] = random.randint(self.MAX_TURN_CHASING // 2, self.MAX_TURN_CHASING)

        # 目標未設定の場合、新たに目標を設定する
        target_candidates = deque()
        for tx in range(MAX_X):
            for ty in range(MAX_Y):
                if pet_distance[tx][ty] == self.RADIUS and self.room[tx][ty] == SPACE:
                    target_candidates.append((tx, ty))
        for hi in range(self.M):
            if (-1, -1) == self.targets[hi]:
                while target_candidates:
                    tx, ty = target_candidates.pop()
                    self.targets[hi] = (tx, ty)

    
    def get_op(self, hi: int) -> str:
        """次の1手を選ぶ。targetに柵を設置しようと試み、できなければ移動する。"""
        hx = self.humen[hi].hx
        hy = self.humen[hi].hy
        op = "."
        distance = self.bfs_all(self.room, hx, hy)
        
        tx, ty = self.targets[hi]
        dx = tx - hx
        dy = ty - hy
        if abs(dx) + abs(dy) == 1 and self.is_puttable(self.room, hi, tx, ty):
            op = DIST2PUT[(dx, dy)]
        else:
            ops = self.get_ops_moveto(distance, tx, ty)
            op = ops[0] if len(ops) else "."
        return op


def read_init() -> Tuple[int, List[Pet], int, List[Human]]:
    N = int(input().strip())
    PETS = []
    for i in range(N):
        px, py, pt = map(int, input().strip().split())
        PETS.append(Pet(px, py, pt))
    M = int(input().strip())
    HUMEN = []
    for i in range(M):
        hx, hy = map(int, input().strip().split())
        HUMEN.append(Human(hx, hy))
    return (N, PETS, M, HUMEN)


def read(N: int, pets: List[Pet], strategy: Strategy) -> None:
    op_pets = list(input().strip().split())
    for pi in range(N):
        pets[pi].update(op_pets[pi])
    strategy.update()


def write(M: int, humen: List[Human]) -> None:
    print("".join([human.next_op for human in humen]), flush=True)
    for hi in range(M):
        humen[hi].update()


def init_room() -> List[List[int]]:
    """部屋を初期化する"""
    room = [[SPACE for y in range(MAX_Y)] for x in range(MAX_X)]
    for y in range(MAX_Y):
        room[0][y] = FENCE
        room[MAX_X-1][y] = FENCE
    for x in range(MAX_X):
        room[x][0] = FENCE
        room[x][MAX_Y-1] = FENCE
    return room


def is_separated(room, pet: Pet, humen: List[Human]) -> bool:
    """人の領域とターゲットの領域が別ならTrue"""
    distance = [[-1 for y in range(MAX_Y)] for x in range(MAX_X)]

    def bfs(sx, sy) -> bool:
        q = deque()
        q.append((sx, sy))
        distance[sx][sy] = 0
        while q:
            x, y = q.popleft()
            if (x, y) in [(human.hx, human.hy) for human in humen]:
                return False
            for op_move in UDLR_MOVE:
                dx, dy = OP2DIST[op_move]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < MAX_X and 0 <= ny < MAX_Y):
                    continue
                elif distance[nx][ny] != -1:
                    continue
                elif room[nx][ny] == SPACE:
                    distance[nx][ny] = distance[x][y] + 1
                    q.append((nx, ny))
        return True
    
    px = pet.px
    py = pet.py
    return bfs(px, py)


def solve(seed, T=300, *args, **kwargs):
    random.seed(seed)
    N, pets, M, humen = read_init()

    room = init_room()
    # 人の行動戦略
    strategy = BuildTrapStrategy(room, N, pets, M, humen)
    
    for k in range(T):
        if isinstance(strategy, BuildTrapStrategy) and strategy.is_terminate():
            strategy = ActivateTrapStrategy(room, N, pets, M, humen)
        if isinstance(strategy, ActivateTrapStrategy) and strategy.is_terminate():
            strategy = CrossStrategy(room, N, pets, M, humen)
        if isinstance(strategy, CrossStrategy) and strategy.is_terminate():
            strategy = CloseCrossStrategy(room, N, pets, M, humen)
        for hi in range(M):
            op = strategy.get_op(hi)
            humen[hi].set_op(room, op)
        write(M, humen)
        read(N, pets, strategy)
        
    logger.error(pets)


def parse_arguments():
    parser = argparse.ArgumentParser(description="AHC008 solver.")
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO)
    s = time.time()
    solve(**vars(args))
    t = time.time()
    logger.info(f"time: {t - s}")
    logger.info(args)
