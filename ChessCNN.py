import random
import time
import tkinter as tk
import tkinter.messagebox as messagebox

import chess
import chess.engine
import numpy as np

import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
from keras.models import load_model

BOARD_SIZE = 8
SQUARE_SIZE = 60
COLOR_LIGHT_SQUARE = "#FFCE9E"
COLOR_DARK_SQUARE = "#D18B47"


def random_board_generator(max_depth):
    chess_board = chess.Board()
    depth = random.randrange(0, max_depth)
    for i in range(depth):
        all_moves = list(chess_board.legal_moves)
        random_move = random.choice(all_moves)
        chess_board.push(random_move)
        if chess_board.is_game_over():
            break
    return chess_board

def stockfish_evaluation(board, depth):
    with chess.engine.SimpleEngine.popen_uci('C:\Stockfish\stockfish') as stockfish:
        result = stockfish.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].black().score()
    return score

board_positions = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}

def square_to_index(square):
    letter = chess.square_name(square)
    row = 8 - int(letter[1])
    column = board_positions[letter[0]]
    return row, column

def board_to_matrix(board):
    board_3d = np.zeros((14, 8, 8), dtype=np.int8)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            index = np.unravel_index(square, (8, 8))
            board_3d[piece - 1][7 - index[0]][index[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            index = np.unravel_index(square, (8, 8))
            board_3d[piece + 5][7 - index[0]][index[1]] = 1
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[13][i][j] = 1
    board.turn = aux
    return board_3d

def populate_dataset(size):
    b = []
    v = []
    for i in range(size):
        new_board = random_board_generator(200)
        matrix_representation = board_to_matrix(new_board)
        valuation = stockfish_evaluation(new_board, 10)
        if valuation is None:
            continue
        b.append(matrix_representation)
        v.append(valuation)
    b = np.array(b)
    v = np.array(v)
    np.savez('dataset', b=b, v=v)

def build_conv_model(conv_size, conv_depth):
    board_3d = layers.Input(shape=(14, 8, 8))
    x = board_3d
    for i in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)
    return models.Model(inputs=board_3d, outputs=x)

def get_dataset():
    container = np.load('dataset.npz')
    b, v = container['b'], container['v']
    v = np.asarray(v / abs(v).max() / 2 + 0.5, dtype=np.float32)
    return b, v

def train_model(model, x_train, y_train):
    model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
    model.summary()
    model.fit(
        x_train, y_train,
        batch_size=2048,
        epochs=2,
        verbose=1,
        validation_split=0.1,
        callbacks=[
            callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
            callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)
        ]
    )
    model.save('chess_model.h5')

def minimax_eval(board, model):
    board3d = board_to_matrix(board)
    board3d = np.expand_dims(board3d, 0)
    return model.predict(board3d)[0][0]

def minimax(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return minimax_eval(board, model)
    if maximizing_player:
        max_eval = -np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = np.inf
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_ai_move(board, depth, model):
    max_move = None
    max_eval = -np.inf
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False, model)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            max_move = move
    return max_move

calculating_label = None

def create_chessboard(board):
    root = tk.Tk()
    root.title("Chess Game")
    canvas = tk.Canvas(root, width=SQUARE_SIZE * BOARD_SIZE + 40, height=SQUARE_SIZE * BOARD_SIZE + 40)
    canvas.pack()
    highlighted_squares = []

    def highlight_square(row, col, color):
        x0, y0 = col * SQUARE_SIZE + 20, row * SQUARE_SIZE + 20
        x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE
        canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=4)
        highlighted_squares.append((x0, y0, x1, y1))

    def draw_chessboard():
        outer_padding = 20
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = COLOR_LIGHT_SQUARE if (row + col) % 2 == 0 else COLOR_DARK_SQUARE
                x0 = outer_padding + col * (SQUARE_SIZE)
                y0 = outer_padding + row * (SQUARE_SIZE)
                x1 = x0 + SQUARE_SIZE
                y1 = y0 + SQUARE_SIZE
                canvas.create_rectangle(x0, y0, x1, y1, fill=color)
                piece = board.piece_at(chess.square(col, 7 - row))
                if piece is not None:
                    color = "white" if piece.color == chess.WHITE else "black"
                    text = piece.unicode_symbol()
                    font = ("Helvetica", 32)
                    text_x = x0 + SQUARE_SIZE // 2
                    text_y = y0 + SQUARE_SIZE // 2
                    canvas.create_text(text_x, text_y, text=text, fill=color, font=font)

        for i in range(BOARD_SIZE):
            x = outer_padding // 2
            y = outer_padding + (BOARD_SIZE - i - 1) * (SQUARE_SIZE) + SQUARE_SIZE // 2
            canvas.create_text(x, y, text=str(i + 1), fill="#FEF8F2")

        for i in range(BOARD_SIZE):
            x = outer_padding + BOARD_SIZE * SQUARE_SIZE + outer_padding // 2
            y = outer_padding + (BOARD_SIZE - i - 1) * (SQUARE_SIZE) + SQUARE_SIZE // 2
            canvas.create_text(x, y, text=str(i + 1), fill="#FEF8F2")

        for i, letter in enumerate("abcdefgh"):
            x = outer_padding + i * (SQUARE_SIZE) + SQUARE_SIZE // 2
            y = outer_padding // 2
            canvas.create_text(x, y, text=letter, fill="#FEF8F2")

        for i, letter in enumerate("abcdefgh"):
            x = outer_padding + i * (SQUARE_SIZE) + SQUARE_SIZE // 2
            y = outer_padding + BOARD_SIZE * SQUARE_SIZE + outer_padding // 2
            canvas.create_text(x, y, text=letter, fill="#FEF8F2")

    def handle_user_move():
        global calculating_label
        user_move = entry.get()
        clear_highlight()
        update_board_display()
        try:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
                update_board_display()

                if board.is_checkmate():
                    messagebox.showinfo("Game Over", "Checkmate! You've won the game.")
                elif board.is_check():
                    messagebox.showinfo("Check", "AI in check!")

                if calculating_label:
                    calculating_label.config(text="AI is calculating...")
                else:
                    calculating_label = tk.Label(root, text="AI is calculating...")
                    calculating_label.pack()

                start_time = time.time()
                canvas.after(50, lambda: [
                    board.push(get_ai_move(board, 2, model)),
                    update_board_display(),
                    messagebox.showinfo("Game Over",
                                        "Checkmate! You've lost the game.") if board.is_checkmate() else None,
                    messagebox.showinfo("Check", "You are in check.") if board.is_check() else None,
                    calculating_label.config(text=f"AI move calculated in {round(time.time() - start_time, 2)} seconds")
                ])

                update_board_display()
            else:
                messagebox.showwarning("Invalid Move", "Invalid move. Please try again.")
                highlight_invalid_move(user_move)
        except ValueError:
            messagebox.showerror("Invalid Move Format",
                                 "Invalid move format. Please enter a move in UCI notation (e.g., e2e4).")
        entry.delete(0, tk.END)

    def highlight_invalid_move(user_move):
        try:
            move = chess.Move.from_uci(user_move)
            from_square = move.from_square
            to_square = move.to_square
            from_row, from_col = 7 - from_square // 8, from_square % 8
            to_row, to_col = 7 - to_square // 8, to_square % 8
            highlight_square(from_row, from_col, "red")
            highlight_square(to_row, to_col, "red")
        except ValueError:
            messagebox.showerror("Invalid Move Format",
                                 "Invalid move format. Please enter a move in UCI notation (e.g., e2e4).")

    def clear_highlight():
        for square in highlighted_squares:
            canvas.create_rectangle(square, fill="", outline="")
        del highlighted_squares[:]

    def update_board_display():
        canvas.delete("all")
        draw_chessboard()

    label = tk.Label(root, text="Enter your move (e.g., e2e4):")
    label.pack()

    entry = tk.Entry(root)
    entry.pack()

    button = tk.Button(root, text="Make Move", command=handle_user_move)
    button.pack()

    draw_chessboard()

    root.mainloop()

if __name__ == "__main__":
    try:
        model = load_model('chess_model.h5')
        print("Using existing model for game.")
    except FileNotFoundError:
        print("Model not found. Training a new model...")
        populate_dataset(1000)
        x_train, y_train = get_dataset()
        model = build_conv_model(32, 4)
        train_model(model, x_train, y_train)

    board = chess.Board()
    create_chessboard(board)
