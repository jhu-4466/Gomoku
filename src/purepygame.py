import pygame
import sys
import math



# Screen attributes
SCREEN_WIDTH = 780
SCREEN_HEIGHT = 820
MENU_HEIGHT = 40

# Board attributes
GRID_SIZE = 19
CELL_SIZE = 40
BOARD_MARGIN = 30
BOARD_WIDTH = CELL_SIZE * (GRID_SIZE - 1)
BOARD_HEIGHT = BOARD_WIDTH
BOARD_RECT = pygame.Rect(
    BOARD_MARGIN,
    OARD_MARGIN + MENU_HEIGHT,
    BOARD_WIDTH,
    BOARD_HEIGHT
)


# Colors
COLOR_BOARD = (230, 200, 150)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_LINE = (50, 50, 50)
COLOR_MENU_BG = (60, 60, 60)
COLOR_DROPDOWN_BG = (70, 70, 70)
COLOR_TEXT = (220, 220, 220)
COLOR_MENU_HOVER_BG = (85, 85, 85)
COLOR_WINNER_BG = (0, 0, 0, 180)



# Star points

STAR_POINTS = [
    (3, 3), (3, 9), (3, 15), 
    (9, 3), (9, 9), (9, 15), 
    (15, 3), (15, 9), (15, 15)
]



"""

--- Gomoku Main APP ---

"""

class Gomoku:

    def __init__(self):

        pygame.init()

        pygame.display.set_caption("Gomoku(Five in a Row)")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.font_small = pygame.font.SysFont("sans", 20)

        self.font_large = pygame.font.SysFont("sans", 50)

        self.clock = pygame.time.Clock()



        self.is_dialog_active = False

        self.dialog_elements = {}



        self.menus = {}

        self.active_menu = None  # Selected dropdown menu

        self.setup_menus()

       

        self.reset_game()



    def setup_menus(self):

        # Menus and actions mapping

        menu_action_map = {

            "Game": [

                {"text": "New Game", "action": self.reset_game},

                {"text": "Undo", "action": self.undo_move},

                {"text": "Exit", "action": sys.exit},

            ],

            "Help": [

                {"text": "Rule", "action": self.rule_description},

            ]

        }

       

        """

        Render Menus

        """

        # for dropdown menus

        global_max_width = -math.inf

        for dropdown_items in menu_action_map.values():

            for item in dropdown_items:

                item_width = self.font_small.render(item["text"], True, COLOR_TEXT).get_width()

                if item_width > global_max_width:

                    global_max_width = item_width



        x_pos = 15

        for menu_name in menu_action_map.keys():

            # main menu item

            rect = pygame.Rect(x_pos, 5, self.font_small.render(menu_name, True, COLOR_TEXT).get_width() + 20, 30)

            self.menus[menu_name] = {

                "rect": rect,

                "items": menu_action_map[menu_name],

                "hover": False,

                "item_surfs": [] # dropdown items

            }

           

            """

            Render Dropdown Items

            """

            y_offset = rect.bottom + 5

            for item in self.menus[menu_name]["items"]:

                item_rect = pygame.Rect(rect.left, y_offset, global_max_width + 20, 30)

                item["rect"] = item_rect

                item["hover"] = False

                y_offset += 30



            x_pos += rect.width + 5



    def open_rules_dialog(self):

        if self.is_dialog_active: return



        self.is_dialog_open = True

        rules_text = [

            "Gomoku Rules",

            "",

            "1. Two players (Black and White) take turns.",

            "2. Black always moves first.",

            "3. Place a stone on an empty intersection.",

            "4. The first player to form an unbroken line",

            "   of five stones wins the game.",

            "5. The line can be horizontal, vertical,",

            "   or diagonal.",

            "6. Once placed, stones cannot be moved."

        ]

       

        # render

        dialog_width, dialog_height = 600, 400

        dialog_x = (SCREEN_WIDTH - dialog_width) / 2

        dialog_y = (SCREEN_HEIGHT - dialog_height) / 2

        self.dialog_elements['bg_rect'] = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)



        self.dialog_elements['text_surfs'] = []

        y_offset = dialog_y + 20

        for i, line in enumerate(rules_text):

            font = self.font_large if i == 0 else self.font_dialog

            color = (255, 255, 0) if i == 0 else COLOR_TEXT

            surf = font.render(line, True, color)

            self.dialog_elements['text_surfs'].append((surf, y_offset))

            y_offset += 45 if i == 0 else 30

           

        btn_width, btn_height = 100, 40

        btn_x = self.dialog_elements['bg_rect'].centerx - (btn_width / 2)

        btn_y = self.dialog_elements['bg_rect'].bottom - btn_height - 20

        self.dialog_elements['close_btn_rect'] = pygame.Rect(btn_x, btn_y, btn_width, btn_height)

        self.dialog_elements['close_btn_hover'] = False



    def reset_game(self):

        self.board = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

        self.current_player = 1

        self.game_over = False

        self.winner = None

        self.move_history = []



    def run(self):

        while True:

            self.handle_events()

            self.draw()

            self.clock.tick(60)



    def handle_events(self):

        mouse_pos = pygame.mouse.get_pos()

       

        """

        Update Hover States

        """

        # Main menu

        for menu_name, menu_data in self.menus.items():

            menu_data["hover"] = menu_data["rect"].collidepoint(mouse_pos)

        # Dropdown items

        if self.active_menu:

            for item in self.menus[self.active_menu]["items"]:

                if item.get("type") != "separator":

                    item["hover"] = item["rect"].collidepoint(mouse_pos)



        for event in pygame.event.get():

            if event.type == pygame.QUIT:

                sys.exit()



            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

                clicked_on_dropdown_item = False

               

                # if click on dropdown menu item

                if self.active_menu:

                    for item in self.menus[self.active_menu]["items"]:

                        if item["hover"]:

                            item["action"]()  # to trigger the mapping action

                            clicked_on_dropdown_item = True

                            self.active_menu = None # to close the dropdown

                            break

                # if not click on dropdown menu item, means click on other menu or board

                # here handle the case where the user clicks on the main menu bar

                if not clicked_on_dropdown_item:

                    currently_active = self.active_menu

                    self.active_menu = None

                    for menu_name, menu_data in self.menus.items():

                        if menu_data["hover"]:

                            if menu_name != currently_active:

                                self.active_menu = menu_name

                            clicked_on_dropdown_item = True

                            break

               

                # click on the board

                if not clicked_on_dropdown_item and not self.game_over:

                    if BOARD_RECT.collidepoint(mouse_pos):

                        self.handle_board_click(mouse_pos)

                       

    def handle_board_click(self, mouse_pos):

        x, y = mouse_pos

        row = round((y - BOARD_RECT.top) / CELL_SIZE)

        col = round((x - BOARD_RECT.left) / CELL_SIZE)

       

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE and self.board[row][col] == 0:

            self.board[row][col] = self.current_player

            self.move_history.append((row, col))

            if self.check_win(row, col):

                self.game_over = True

                self.winner = self.current_player

            else:

                self.current_player = 3 - self.current_player



    def undo_move(self):

        if self.move_history and not self.game_over:

            last_row, last_col = self.move_history.pop()

            self.board[last_row][last_col] = 0

            self.current_player = 3 - self.current_player



    def rule_description(self):

        print("Gomoku Rule:")

        print("1. Two players take turns placing stones on a 19x19 board.")

        print("2. One player uses black stones, the other uses white.")

        print("3. Black goes first.")

        print("4. The goal is to be the first to form an unbroken line of five stones.")

        print("5. The line can be horizontal, vertical, or diagonal.")

        print("6. Overlines (more than five in a row) still count as a win.")

        print("7. No pieces can be moved once placed.")





    def check_win(self, row, col):

        player = self.board[row][col]

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for d_row, d_col in directions:

            count = 1

            for i in range(1, 5):

                r, c = row + i * d_row, col + i * d_col

                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and self.board[r][c] == player: count += 1

                else: break

            for i in range(1, 5):

                r, c = row - i * d_row, col - i * d_col

                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and self.board[r][c] == player: count += 1

                else: break

            if count >= 5: return True

        return False



    def draw(self):

        self.screen.fill(COLOR_BOARD)

        self.draw_board_grid()

        self.draw_pieces()

        self.draw_menus()

        if self.game_over:

            self.draw_winner_message()

        pygame.display.flip()



    def draw_menus(self):

        pygame.draw.rect(self.screen, COLOR_MENU_BG, (0, 0, SCREEN_WIDTH, MENU_HEIGHT))



        # Main menu

        for menu_name, menu_data in self.menus.items():

            if self.active_menu == menu_name or menu_data["hover"]:

                pygame.draw.rect(self.screen, COLOR_MENU_HOVER_BG, menu_data["rect"], border_radius=4)

            text_surf = self.font_small.render(menu_name, True, COLOR_TEXT)

            text_rect = text_surf.get_rect(center=menu_data["rect"].center)

            self.screen.blit(text_surf, text_rect)



        # Dropdown items

        if self.active_menu:

            menu_data = self.menus[self.active_menu]

            first_item_rect = menu_data["items"][0]["rect"]

            last_item_rect = menu_data["items"][-1]["rect"]

            dropdown_bg_rect = pygame.Rect(first_item_rect.left, first_item_rect.top, first_item_rect.width, last_item_rect.bottom - first_item_rect.top)

            pygame.draw.rect(self.screen, COLOR_DROPDOWN_BG, dropdown_bg_rect.inflate(2,2), border_radius=4)

           

            for item in menu_data["items"]:

                if item["hover"]:

                    pygame.draw.rect(self.screen, COLOR_MENU_HOVER_BG, item["rect"], border_radius=4)



                text_surf = self.font_small.render(item["text"], True, COLOR_TEXT)

                text_rect = text_surf.get_rect(centery=item["rect"].centery, left=item["rect"].left + 10)

                self.screen.blit(text_surf, text_rect)



    def draw_board_grid(self):

        for i in range(GRID_SIZE):

            start_pos_v = (BOARD_RECT.left + i * CELL_SIZE, BOARD_RECT.top)

            end_pos_v = (BOARD_RECT.left + i * CELL_SIZE, BOARD_RECT.bottom)

            pygame.draw.line(self.screen, COLOR_LINE, start_pos_v, end_pos_v)

            start_pos_h = (BOARD_RECT.left, BOARD_RECT.top + i * CELL_SIZE)

            end_pos_h = (BOARD_RECT.right, BOARD_RECT.top + i * CELL_SIZE)

            pygame.draw.line(self.screen, COLOR_LINE, start_pos_h, end_pos_h)

        for r, c in STAR_POINTS:

            pos = (BOARD_RECT.left + c * CELL_SIZE, BOARD_RECT.top + r * CELL_SIZE)

            pygame.draw.circle(self.screen, COLOR_LINE, pos, 6)



    def draw_pieces(self):

        for r in range(GRID_SIZE):

            for c in range(GRID_SIZE):

                player = self.board[r][c]

                if player != 0:

                    pos = (BOARD_RECT.left + c * CELL_SIZE, BOARD_RECT.top + r * CELL_SIZE)

                    color = COLOR_BLACK if player == 1 else COLOR_WHITE

                    pygame.draw.circle(self.screen, color, pos, CELL_SIZE // 2 - 2)



    def draw_winner_message(self):

        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

        overlay.fill(COLOR_WINNER_BG)

        self.screen.blit(overlay, (0, 0))

        winner_text = "Black Wins!" if self.winner == 1 else "White Wins!"

        text_surf = self.font_large.render(winner_text, True, COLOR_WHITE)

        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))

        self.screen.blit(text_surf, text_rect)



if __name__ == '__main__':

    game = Gomoku()

    game.run()