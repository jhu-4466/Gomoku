# 这可以是一个新文件, e.g., 'gomoku_ai.py', 或者直接集成到你的主代码中


class NegaMaxAI:
    def __init__(self, grid_size=19):
        self.GRID_SIZE = grid_size
        # 定义棋型分数 (这些权重就是未来“训练”或优化的对象)
        self.PATTERN_SCORES = {
            "FIVE": 100000,
            "LIVE_FOUR": 10000,
            "RUSH_FOUR": 800,
            "LIVE_THREE": 500,
            "SLEEP_THREE": 100,
            "LIVE_TWO": 50,
            "SLEEP_TWO": 10,
        }

    def evaluate_board(self, board, player):
        player_score = self._calculate_score_for_player(board, player)
        opponent_score = self._calculate_score_for_player(board, 3 - player)

        # 核心：返回我方分数与对手分数的差值
        return player_score - opponent_score

    def _calculate_score_for_player(self, board, player):
        total_score = 0
        # 遍历所有可能的五元组 (horizontal, vertical, two diagonals)
        # 并根据棋型计分
        # ... 实现扫描和计分的具体逻辑 ...
        # 这是一个复杂但关键的实现，需要仔细检查边界和棋型定义
        # (为简化，此处省略具体实现，但其目标是填充total_score)

        # 伪代码逻辑:
        # for direction in [horizontal, vertical, diag1, diag2]:
        #     for line in all_lines_in_direction:
        #         score += self.evaluate_line(line, player)

        return total_score

    def find_best_move(self, board, depth):
        """
        AI的入口函数：找到最佳落子位置
        """
        best_move = None
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        # 获取所有可以落子的位置
        possible_moves = []
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if board[r][c] == 0:
                    possible_moves.append((r, c))

        # 遍历所有可能的下一步
        for move in possible_moves:
            r, c = move
            # 1. 尝试落子
            board[r][c] = 1  # 假设AI是玩家1

            # 2. 调用negamax进行递归搜索，注意符号取反
            score = -self._negamax(board, depth - 1, -beta, -alpha, 2)  # 切换到对手视角

            # 3. 撤销落子
            board[r][c] = 0

            # 4. 更新最佳分数和最佳落子
            if score > best_score:
                best_score = score
                best_move = move

            # 更新alpha (alpha是我们的下限)
            alpha = max(alpha, best_score)

        return best_move

    def _negamax(self, board, depth, alpha, beta, player):
        """
        Negamax 核心递归函数
        """
        # 1. 到达搜索深度或游戏结束，返回局面评价值
        # (这里可以加入游戏是否胜利的判断)
        if depth == 0:
            return self.evaluate_board(board, player)

        best_score = -float("inf")

        possible_moves = []  # ... 获取所有可落子位置 ...

        for move in possible_moves:
            r, c = move
            board[r][c] = player

            score = -self._negamax(board, depth - 1, -beta, -alpha, 3 - player)

            board[r][c] = 0  # 撤销

            best_score = max(best_score, score)
            alpha = max(alpha, best_score)

            # 关键的Alpha-Beta剪枝
            if alpha >= beta:
                break  # 剪枝！

        return best_score
