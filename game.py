import math
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

HERE = Path(__file__).parent

classes = [0, 1, 2, 3]

# ============================================================
# 1. LOAD DATASET AND PRECOMPUTED STATS
# ============================================================
# Run precompute_stats.py first to create the two CSV files.

with open(HERE / "nuclease_design_processed.pkl", "rb") as f:
    data = pickle.load(f)

X_train = np.asarray(data["X_train_int"])
y_train = np.asarray(data["y_train"]).flatten()

print("Training data loaded:", X_train.shape)
print("Labels loaded:", y_train.shape)

sequence_length = X_train.shape[1]
vocab_size = int(X_train.max()) + 1

mode_csv = HERE / "overall_mode.csv"
freq_csv = HERE / "class_value_freq.csv"

if not mode_csv.exists() or not freq_csv.exists():
    raise FileNotFoundError(
        "Missing overall_mode.csv / class_value_freq.csv. "
        "Run precompute_stats.py first."
    )

# overall_mode[j] = most common value at position j
mode_rows = np.loadtxt(mode_csv, delimiter=",", skiprows=1, dtype=int, ndmin=2)
overall_mode = np.zeros(sequence_length, dtype=int)
overall_mode[mode_rows[:, 0]] = mode_rows[:, 1]

if len(mode_rows) != sequence_length:
    raise ValueError(
        "overall_mode.csv doesn't match the dataset's sequence length. "
        "Re-run precompute_stats.py."
    )

# class_value_freq[c, j, v] = fraction of class c with value v at position j
freq_rows = np.loadtxt(freq_csv, delimiter=",", skiprows=1, ndmin=2)
class_value_freq = np.zeros((len(classes), sequence_length, vocab_size))
class_value_freq[
    freq_rows[:, 0].astype(int),
    freq_rows[:, 1].astype(int),
    freq_rows[:, 2].astype(int),
] = freq_rows[:, 3]


# class_dev_rate[c, j] = fraction of class c that deviates from the overall mode
# at position j (1 minus how often the class has the mode value there).
class_dev_rate = 1.0 - class_value_freq[:, np.arange(sequence_length), overall_mode]

# Real deviation rates are often small, which would leave every bar faint.
# Rescale so the 95th-percentile rate maps to full opacity, then take a square
# root so small-but-real differences are still visible.
_rate_ref = max(np.percentile(class_dev_rate, 95), 1e-9)
class_dev_opacity = np.clip(class_dev_rate / _rate_ref, 0, 1) ** 0.5


# ============================================================
# 2. GAME STATE AND SETTINGS
# ============================================================

colors = {
    0: "#3498db",  # Blue
    1: "#2ecc71",  # Green
    2: "#e67e22",  # Orange
    3: "#9b59b6",  # Purple
}

NEUTRAL_COLOR = "#eeeeee"
AREA_MAX_HEIGHT = 0.9   # leave a little room at the top for the index label
ALPHA_MIN = 0.40        # faintest a bar can be (class almost never deviates here)

rng = np.random.default_rng()

score = 0
total_guesses = 0
current_index = None
answered = False

confusion_matrix = np.zeros((4, 4), dtype=int)   # rows = true class, cols = guessed class

variability_axes = None
variability_highlights = []
fig_var = None
fig_met = None
ax_tbl = None


# ============================================================
# 3. CREATE GAME WINDOW
# ============================================================

ncols = 17
nrows = math.ceil(sequence_length / ncols)

fig = plt.figure(figsize=(16, 9))

ax = fig.add_axes([0.05, 0.25, 0.65, 0.65])
ax_cm = fig.add_axes([0.77, 0.38, 0.20, 0.40])   # confusion matrix, next to the game

plt.subplots_adjust(
    bottom=0.22,
    top=0.86,
    left=0.04,
    right=0.96
)

ax.set_xlim(0, ncols)
ax.set_ylim(0, nrows)
ax.set_aspect("equal")
ax.axis("off")

fig.suptitle(
    "PROTEIN SEQUENCE CLASSIFICATION GAME",
    fontsize=18,
    fontweight="bold"
)

score_text = fig.text(0.5, 0.91, "Score: 0 / 0", ha="center", fontsize=13)

instruction_text = fig.text(
    0.5, 0.16,
    "Which class does this sequence belong to?",
    ha="center",
    fontsize=13,
    fontweight="bold"
)

feedback_text = fig.text(0.5, 0.12, "", ha="center", fontsize=12)


# ============================================================
# 4. DRAW SEQUENCE
# ============================================================

def draw_deviation_area(x0, y0, j, value):
    """One bar per class: height = frequency of this value, opacity = how often the class deviates here."""
    freqs = class_value_freq[:, j, value]
    peak = freqs.max() if freqs.max() > 0 else 1.0   # tallest class fills the cell
    heights = freqs / peak * AREA_MAX_HEIGHT

    n = len(classes)
    bar_w = 1 / n

    # white background so the bars read clearly
    ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor="white", edgecolor="none"))

    for c in classes:
        # Transparency: how often this class deviates from the mode at this position
        alpha = ALPHA_MIN + (1 - ALPHA_MIN) * class_dev_opacity[c, j]
        ax.add_patch(Rectangle(
            (x0 + c * bar_w, y0),
            bar_w,
            heights[c],
            facecolor=to_rgba(colors[c], alpha),   # translucent fill...
            edgecolor=colors[c],                   # ...with a solid outline so it never vanishes
            linewidth=0.6
        ))


def draw_sequence():
    global current_index

    # Select a class uniformly, then a random sequence from that class
    selected_class = rng.choice(classes)
    class_indices = np.where(y_train == selected_class)[0]
    current_index = int(rng.choice(class_indices))

    sequence = X_train[current_index]

    ax.clear()
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.axis("off")

    # White outline keeps text readable over any color
    outline = [pe.withStroke(linewidth=2.5, foreground="white")]

    for j, value in enumerate(sequence):

        col = j % ncols
        row = nrows - 1 - (j // ncols)
        value = int(value)

        if value == overall_mode[j]:
            ax.add_patch(Rectangle((col, row), 1, 1,
                                   facecolor=NEUTRAL_COLOR, edgecolor="none"))
        else:
            draw_deviation_area(col, row, j, value)

        # Cell border
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor="none",
                               edgecolor="black", linewidth=0.7))

        # Value
        ax.text(col + 0.5, row + 0.5, f"{value}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="#222222", path_effects=outline)

        # Position index
        ax.text(col + 0.5, row + 0.83, f"{j}",
                ha="center", va="center", fontsize=6,
                color="#222222", path_effects=outline)

    ax.set_title(f"Sequence #{current_index}", fontsize=12, pad=12)

    fig.canvas.draw_idle()

    # Refresh the outlines in the variability window for this sequence
    if variability_axes is not None:
        update_variability_highlights()


# ============================================================
# 5. HANDLE GUESSES
# ============================================================

def make_guess(c):
    global score, total_guesses, answered

    if answered:
        return

    true_class = int(y_train[current_index])
    confusion_matrix[true_class, c] += 1

    total_guesses += 1

    if c == true_class:
        score += 1
        feedback_text.set_text(
            f"CORRECT! This sequence belongs to Class {true_class}."
        )
        feedback_text.set_color("green")
    else:
        feedback_text.set_text(
            f"INCORRECT! The correct answer was Class {true_class}."
        )
        feedback_text.set_color("red")

    score_text.set_text(
        f"Score: {score} / {total_guesses} "
        f"({100 * score / total_guesses:.1f}%)"
    )

    answered = True
    instruction_text.set_text("Click 'Next Sequence' to continue.")

    fig.canvas.draw_idle()

    update_metrics()


# ============================================================
# 6. VARIABILITY WINDOW
# ============================================================

def show_class_variability():
    global fig_var, variability_axes

    number_per_class = 20

    fig_var, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes = axes.flatten()
    variability_axes = axes

    fig_var.suptitle("Sequence Variability by Class", fontsize=16)

    for ax_var, c in zip(axes, classes):

        class_data = X_train[y_train == c]

        sampled_indices = rng.choice(
            len(class_data),
            size=number_per_class,
            replace=False
        )
        sampled = class_data[sampled_indices]

        modes = np.array([
            np.bincount(sampled[:, j], minlength=20).argmax()
            for j in range(sampled.shape[1])
        ])

        disagreement = np.sum(sampled != modes, axis=0)
        opacity = disagreement / number_per_class

        ax_var.clear()
        ax_var.set_xlim(0, ncols)
        ax_var.set_ylim(0, nrows)
        ax_var.set_aspect("equal")
        ax_var.axis("off")

        for j, value in enumerate(modes):

            col = j % ncols
            row = nrows - 1 - (j // ncols)

            # Class color, with opacity based on disagreement
            ax_var.add_patch(Rectangle(
                (col, row), 1, 1,
                facecolor=colors[c],
                alpha=opacity[j],
                edgecolor="black",
                linewidth=0.7
            ))

            ax_var.text(col + 0.5, row + 0.83, str(j),
                        ha="center", va="center", fontsize=6, color="black")

            ax_var.text(col + 0.5, row + 0.5, str(value),
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color="black")

        ax_var.set_title(f"Class {c}", fontsize=11)

    plt.tight_layout()


def update_variability_highlights():
    global variability_highlights

    if variability_axes is None or current_index is None:
        return

    # Remove every outline from the previous sequence
    for border in variability_highlights:
        border.remove()
    variability_highlights.clear()

    current_sequence = X_train[current_index]

    # Outline the positions that deviate from the overall mode
    for ax_var in variability_axes:
        for j, value in enumerate(current_sequence):

            if int(value) != overall_mode[j]:
                col = j % ncols
                row = nrows - 1 - (j // ncols)

                border = Rectangle(
                    (col, row), 1, 1,
                    facecolor="none",
                    edgecolor="cyan",
                    linewidth=2.5,
                    zorder=10
                )

                ax_var.add_patch(border)
                variability_highlights.append(border)

    fig_var.canvas.draw()
    fig_var.canvas.flush_events()


# ============================================================
# 7. LIVE METRICS
# ============================================================

def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def update_confusion_matrix():
    """Redraw the confusion matrix in the main window (next to the game)."""
    cm = confusion_matrix
    true_totals = cm.sum(axis=1).astype(float)   # how many of each class you were shown

    ax_cm.clear()
    row_norm = safe_div(cm.astype(float), true_totals[:, None] * np.ones((1, 4)))
    ax_cm.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(4):
        for j in range(4):
            pct = row_norm[i, j] * 100
            ax_cm.text(
                j, i, f"{cm[i, j]}\n({pct:.0f}%)",
                ha="center", va="center", fontsize=9,
                fontweight="bold" if i == j else "normal",
                color="white" if row_norm[i, j] > 0.55 else "black"
            )

    ax_cm.set_xticks(range(4))
    ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(classes)
    ax_cm.set_yticklabels(classes)
    ax_cm.set_xlabel("Your guess")
    ax_cm.set_ylabel("True class")
    ax_cm.set_title("Confusion Matrix\n(% of each true class)", fontsize=11)

    fig.canvas.draw_idle()


def show_metrics_window():
    global fig_met, ax_tbl

    fig_met, ax_tbl = plt.subplots(figsize=(6, 4))
    fig_met.suptitle("Per-Class Metrics", fontsize=14, fontweight="bold")
    update_metrics_table()


def update_metrics_table():
    """Redraw the precision / recall / F1 table in its own window."""
    if fig_met is None:
        return

    cm = confusion_matrix
    tp = np.diag(cm).astype(float)
    true_totals = cm.sum(axis=1).astype(float)
    guess_totals = cm.sum(axis=0).astype(float)

    recall = safe_div(tp, true_totals)
    precision = safe_div(tp, guess_totals)
    f1 = safe_div(2 * precision * recall, precision + recall)
    total = cm.sum()
    accuracy = tp.sum() / total if total > 0 else 0.0

    ax_tbl.clear()
    ax_tbl.axis("off")

    rows = []
    for c in classes:
        rows.append([
            f"Class {c}",
            f"{int(true_totals[c])}",
            f"{precision[c]:.2f}",
            f"{recall[c]:.2f}",
            f"{f1[c]:.2f}",
        ])
    rows.append([
        "Macro avg", f"{int(total)}",
        f"{precision.mean():.2f}", f"{recall.mean():.2f}", f"{f1.mean():.2f}"
    ])

    table = ax_tbl.table(
        cellText=rows,
        colLabels=["", "n", "Precision", "Recall", "F1"],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # color the class labels to match the game, bold the macro row
    for c in classes:
        table[c + 1, 0].set_text_props(color=colors[c], fontweight="bold")
    for col in range(5):
        table[len(rows), col].set_text_props(fontweight="bold")
        table[0, col].set_facecolor("#eeeeee")

    ax_tbl.set_title(
        f"Overall accuracy: {accuracy * 100:.1f}%  ({int(tp.sum())} / {int(total)})"
    )

    fig_met.tight_layout()
    fig_met.canvas.draw_idle()


def update_metrics():
    update_confusion_matrix()
    update_metrics_table()


# ============================================================
# 8. BUTTONS
# ============================================================

button_axes = []
guess_buttons = []

button_width = 0.13
button_gap = 0.015
start_x = 0.20

for i, c in enumerate(classes):

    button_ax = fig.add_axes([
        start_x + i * (button_width + button_gap),
        0.035,
        button_width,
        0.055
    ])

    button = Button(
        button_ax,
        f"Class {c}",
        color=colors[c],
        hovercolor="#cccccc"
    )

    button.label.set_color("white")
    button.label.set_fontweight("bold")
    button.on_clicked(lambda event, c=c: make_guess(c))

    guess_buttons.append(button)
    button_axes.append(button_ax)


# Next sequence button
next_ax = fig.add_axes([0.78, 0.035, 0.15, 0.055])

next_button = Button(
    next_ax,
    "Next Sequence",
    color="#dddddd",
    hovercolor="#bbbbbb"
)


def next_sequence(event):
    global answered

    if not answered:
        return

    answered = False

    feedback_text.set_text("")
    instruction_text.set_text("Which class does this sequence belong to?")

    draw_sequence()


next_button.on_clicked(next_sequence)


# ============================================================
# 9. LEGEND
# ============================================================

legend_text = (
    "Light gray = Overall mode     "
    "Bar height = how often this value appears in each class; opacity = how often that class deviates here     "
    + "     ".join([f"Class {c}" for c in classes])
)

fig.text(0.5, 0.005, legend_text, ha="center", fontsize=9)


# ============================================================
# 10. START GAME
# ============================================================

draw_sequence()
show_class_variability()
show_metrics_window()
update_confusion_matrix()
update_variability_highlights()
plt.show()