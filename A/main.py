import sys
import argparse
import time
import random
import logging
from typing import Tuple, List, Any
from collections import deque

input = sys.stdin.readline
logger = logging.getLogger(__name__)
OP2DIST = {
    "u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1),  # 仕切りを設置
    "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1),  # 移動
    ".": (0, 0)                                            # 何もしない
}
UDLR_MOVE = "UDLR"
UDLR_PUT = "udlr"
REVERSE_OP = {
    "u": "d", "d": "u", "l": "r", "r": "l",  # 仕切りを設置
    "U": "D", "D": "U", "L": "R", "R": "L",  # 移動
    ".": "."                                 # 何もしない
}

SPACE = -2
SPACE_BOOKED = -1
FENCE = 0

MAX_X = 32
MAX_Y = 32

def read_init() -> Tuple[int, List[Any], int, List[Any]]:
    N = int(input().strip())
    PETS = []
    for i in range(N):
        px, py, pt = map(int, input().strip().split())
        PETS.append([px, py, pt])
    M = int(input().strip())
    HUMEN = []
    for i in range(M):
        hx, hy = map(int, input().strip().split())
        HUMEN.append([hx, hy])
    return (N, PETS, M, HUMEN)


def read() -> List[str]:
    ops = list(input().strip().split())
    return ops

def write(ops: List[str]) -> None:
    print("".join(ops), flush=True)


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


def book_fences(room) -> List[List[int]]:
    """仕切りの設置を予約する"""
    for y in range(1, MAX_Y-1):
        room[MAX_X // 2][y] = SPACE_BOOKED
    for x in range(1, MAX_X-1):
        room[x][MAX_Y // 2] = SPACE_BOOKED


def nearest_place_point(room, hi: int, hx: int, hy: int) -> Tuple[int, int, List[str]]:
    """仕切りを設置可能なマスのうち最も近いマスと手順の組を返す"""
    distance = [[-1 for y in range(MAX_Y)] for x in range(MAX_X)]

    def bfs(x, y):
        q = deque()
        q.append((x, y))
        distance[x][y] = 0
        while q:
            x, y = q.popleft()
            for op_move in UDLR_MOVE:
                dx, dy = OP2DIST[op_move]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < MAX_X and 0 <= ny < MAX_Y):
                    continue
                elif distance[nx][ny] != -1:
                    continue
                elif room[nx][ny] == SPACE_BOOKED:
                    return (x, y, op_move.lower())
                elif room[nx][ny] == SPACE:
                    distance[nx][ny] = distance[x][y] + 1
                    q.append((nx, ny))
        return (-1, -1, ".")
    
    tx, ty, op = bfs(hx, hy)

    if tx == -1 and ty == -1:
        return (hx, hy, ["."])
    
    if tx == hx and ty == hy:
        return (hx, hy, [op])
    
    ops = [op]
    def dfs(x, y):
        if distance[x][y] == 0:
            return
        for op_move in UDLR_MOVE:
            dx, dy = OP2DIST[op_move]
            nx, ny = x - dx, y - dy
            if not (0 <= nx < MAX_X and 0 <= ny < MAX_Y):
                continue
            if distance[nx][ny] == distance[x][y] - 1:
                ops.append(op_move)
                dfs(nx, ny)
                break
    
    dfs(tx, ty)
    return (tx, ty, ops[::-1])


def put_fence_if_can(room, N, pets, M, humen, fx: int, fy: int) -> bool:
    """(fx, fx) に仕切りを設置可能なら設置し、Trueを返す"""
    for x, y in humen:
        if x == fx and y == fy:
            return False
    for x, y, t in pets:
        if x == fx and y == fy:
            return False
        if x+1 == fx and y == fy:
            return False
        if x == fx and y+1 == fy:
            return False
        if x-1 == fx and y == fy:
            return False
        if x == fx and y-1 == fy:
            return False
    if room[fx][fy] != SPACE_BOOKED:
        return False
    room[fx][fy] = FENCE
    return True


def update_humen(M, humen, op_humen: List[str]) -> None:
    for i in range(M):
        hx, hy = humen[i]
        for op in op_humen[i]:
            if op == "U":
                hx -= 1
            if op == "D":
                hx += 1
            if op == "L":
                hy -= 1
            if op == "R":
                hy += 1
        humen[i] = [hx, hy]

def update_pets(N, pets, op_pets: List[str]) -> None:
    for i in range(N):
        px, py, pt = pets[i]
        for op in op_pets[i]:
            if op == "U":
                px -= 1
            if op == "D":
                px += 1
            if op == "L":
                py -= 1
            if op == "R":
                py += 1
        pets[i] = [px, py, pt]


def solve(seed, T=300, *args, **kwargs):
    random.seed(seed)
    N, pets, M, humen = read_init()
    L = 0
    room = init_room()
    book_fences(room)
    
    for k in range(T):
        op_humen = ["."] * M
        for hi in range(M):
            hx, hy = humen[hi]
            jx, jy, ops = nearest_place_point(room, hi, hx, hy)
            op = ops[0]
            if random.random() < 0.1:
                op_humen[hi] = "."
            elif op in UDLR_MOVE:
                op_humen[hi] = op
            elif op in UDLR_PUT:
                dx, dy = OP2DIST[op]
                puttable = put_fence_if_can(room, N, pets, M, humen, hx+dx, hy+dy)
                if puttable:
                    op_humen[hi] = op
        write(op_humen)
        op_pets = read()
        update_humen(M, humen, op_humen)
        update_pets(N, pets, op_pets)
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
