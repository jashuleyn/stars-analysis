"""
Star Type Classification — Interactive GUI with ML & Deep Learning
Dataset: https://www.kaggle.com/datasets/brsdincer/star-type-classification
GUI: tkinter with live training charts, metrics, and result visualization
"""

import os, warnings, threading, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ──────────────────────────────────────────────────────────────────
# Constants & Theme
# ──────────────────────────────────────────────────────────────────
TYPE_LABELS = {
    0: 'Brown Dwarf', 1: 'Red Dwarf',   2: 'White Dwarf',
    3: 'Main Seq.',   4: 'Supergiant',  5: 'Hypergiant',
}
STAR_COLORS = {
    0: '#8B4513', 1: '#FF4500', 2: '#E0E0FF',
    3: '#FFD700', 4: '#FFA500', 5: '#4169E1',
}
PALETTE   = ['#6C63FF', '#FF6584', '#43B89C', '#F5A623', '#50C5B7', '#E84393']
BG_DARK   = '#0D1117'
BG_MID    = '#161B22'
BG_PANEL  = '#1C2128'
BORDER    = '#30363D'
FG_BRIGHT = '#E6EDF3'
FG_DIM    = '#8B949E'
ACCENT    = '#6C63FF'

MPL_STYLE = {
    'figure.facecolor': BG_DARK,   'axes.facecolor': BG_MID,
    'axes.edgecolor':   BORDER,    'text.color':     FG_BRIGHT,
    'xtick.color':      FG_DIM,    'ytick.color':    FG_DIM,
    'axes.labelcolor':  FG_BRIGHT, 'grid.color':     '#21262D',
    'grid.alpha':       0.5,       'font.family':    'monospace',
    'axes.spines.top':  False,     'axes.spines.right': False,
}
plt.rcParams.update(MPL_STYLE)

# ──────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ──────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str):
    """Load Stars.csv, encode categoricals, log-transform skewed features."""
    df = pd.read_csv(csv_path)

    df['Color'] = (df['Color'].str.strip()
                               .str.lower()
                               .str.replace('-', ' ', regex=False)
                               .str.replace('  ', ' ', regex=False))
    le_color = LabelEncoder()
    le_spec  = LabelEncoder()
    df['Color_enc']          = le_color.fit_transform(df['Color'])
    df['Spectral_Class_enc'] = le_spec.fit_transform(df['Spectral_Class'])

    features = ['Temperature', 'L', 'R', 'A_M', 'Color_enc', 'Spectral_Class_enc']
    X = df[features].values.astype(float)
    y = df['Type'].values

    # Log-transform right-skewed physical features
    X[:, 0] = np.log1p(X[:, 0])
    X[:, 1] = np.log1p(X[:, 1])
    X[:, 2] = np.log1p(X[:, 2])

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    return df, X_scaled, X_train, X_test, y_train, y_test, features


# ──────────────────────────────────────────────────────────────────
# Neural Network Builder
# ──────────────────────────────────────────────────────────────────
def build_nn(input_dim, n_classes, units, dropout):
    inp = keras.Input(shape=(input_dim,))
    x   = inp
    for u in units:
        x = layers.Dense(u, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    return keras.Model(inp, out)


# ──────────────────────────────────────────────────────────────────
# Live-Update Keras Callback
# ──────────────────────────────────────────────────────────────────
class LiveCallback(keras.callbacks.Callback):
    def __init__(self, gui_ref):
        super().__init__()
        self.gui = gui_ref

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.gui.on_epoch_end(epoch, logs)


# ══════════════════════════════════════════════════════════════════
# Main Application Window
# ══════════════════════════════════════════════════════════════════
class StarClassifierApp(tk.Tk):

    CSV_PATH = 'Stars.csv'   # ← place Stars.csv next to this script

    def __init__(self):
        super().__init__()
        self.title('⭐  Star Type Classifier  —  ML & Deep Learning')
        self.configure(bg=BG_DARK)
        self.geometry('1380x900')
        self.minsize(1100, 750)

        # State
        self.training      = False
        self.train_acc_h   = []
        self.val_acc_h     = []
        self.train_loss_h  = []
        self.val_loss_h    = []
        self.sk_results    = {}
        self.nn_acc        = None
        self.df            = None
        self.nn_model      = None

        self._build_ui()
        self._load_data()

    # ── UI Construction ────────────────────────────────────────────
    def _build_ui(self):
        # ── Title bar ──
        title_bar = tk.Frame(self, bg=BG_DARK)
        title_bar.pack(fill='x', padx=20, pady=(14, 4))

        tk.Label(title_bar, text='⭐  STAR TYPE CLASSIFIER',
                 font=('Courier', 18, 'bold'),
                 fg=ACCENT, bg=BG_DARK).pack(side='left')
        tk.Label(title_bar, text='ML & Deep Learning Pipeline',
                 font=('Courier', 10), fg=FG_DIM, bg=BG_DARK).pack(side='left', padx=12)

        # ── Main paned window: left sidebar | right tabs ──
        paned = tk.PanedWindow(self, orient='horizontal', bg=BORDER,
                                sashwidth=3, sashrelief='flat')
        paned.pack(fill='both', expand=True, padx=8, pady=8)

        # ── LEFT sidebar ──
        left = tk.Frame(paned, bg=BG_PANEL, width=300)
        paned.add(left, minsize=260)
        self._build_sidebar(left)

        # ── RIGHT notebook ──
        right = tk.Frame(paned, bg=BG_DARK)
        paned.add(right, minsize=700)
        self._build_notebook(right)

    def _build_sidebar(self, parent):
        pad = {'padx': 14, 'pady': 4}

        tk.Label(parent, text='DATASET', font=('Courier', 9, 'bold'),
                 fg=ACCENT, bg=BG_PANEL).pack(anchor='w', padx=14, pady=(14, 2))

        self.lbl_dataset = tk.Label(parent, text='Loading…',
                                     font=('Courier', 9), fg=FG_DIM, bg=BG_PANEL,
                                     justify='left', wraplength=260)
        self.lbl_dataset.pack(anchor='w', **pad)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=14, pady=8)

        # ── Model selection ──
        tk.Label(parent, text='MODEL', font=('Courier', 9, 'bold'),
                 fg=ACCENT, bg=BG_PANEL).pack(anchor='w', padx=14)

        self.model_var = tk.StringVar(value='Neural Network (TF)')
        for m in ['Random Forest', 'Gradient Boosting', 'SVM (RBF)', 'Neural Network (TF)']:
            tk.Radiobutton(parent, text=m, variable=self.model_var, value=m,
                           font=('Courier', 9), fg=FG_BRIGHT, bg=BG_PANEL,
                           selectcolor=ACCENT, activebackground=BG_PANEL,
                           activeforeground=FG_BRIGHT).pack(anchor='w', padx=20, pady=1)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=14, pady=8)

        # ── NN hyper-params ──
        tk.Label(parent, text='NEURAL NET PARAMS', font=('Courier', 9, 'bold'),
                 fg=ACCENT, bg=BG_PANEL).pack(anchor='w', padx=14)

        def _row(label, default, var_name):
            f = tk.Frame(parent, bg=BG_PANEL)
            f.pack(fill='x', padx=14, pady=2)
            tk.Label(f, text=label, font=('Courier', 8), fg=FG_DIM,
                     bg=BG_PANEL, width=14, anchor='w').pack(side='left')
            v = tk.StringVar(value=default)
            setattr(self, var_name, v)
            tk.Entry(f, textvariable=v, font=('Courier', 9),
                     fg=FG_BRIGHT, bg=BG_MID, insertbackground=FG_BRIGHT,
                     relief='flat', width=8).pack(side='left')

        _row('Epochs',     '150',      'var_epochs')
        _row('Batch Size', '16',        'var_batch')
        _row('Dropout',    '0.3',       'var_drop')
        _row('Layer 1',    '128',       'var_l1')
        _row('Layer 2',    '64',        'var_l2')
        _row('Layer 3',    '32',        'var_l3')

        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=14, pady=8)

        # ── Train all option ──
        self.var_train_all = tk.BooleanVar(value=False)
        tk.Checkbutton(parent, text='Train ALL models',
                       variable=self.var_train_all,
                       font=('Courier', 9), fg=FG_BRIGHT, bg=BG_PANEL,
                       selectcolor=ACCENT, activebackground=BG_PANEL,
                       activeforeground=FG_BRIGHT).pack(anchor='w', padx=14, pady=2)

        # ── Train button ──
        self.btn_train = tk.Button(
            parent, text='▶  START TRAINING',
            font=('Courier', 11, 'bold'), fg='#0D1117', bg=ACCENT,
            relief='flat', cursor='hand2', activebackground='#8078FF',
            command=self._on_train_click, pady=8)
        self.btn_train.pack(fill='x', padx=14, pady=(10, 4))

        # ── Progress bar ──
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Accent.Horizontal.TProgressbar',
                        troughcolor=BG_MID, background=ACCENT,
                        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
        self.progress = ttk.Progressbar(
            parent, variable=self.progress_var, maximum=100,
            style='Accent.Horizontal.TProgressbar')
        self.progress.pack(fill='x', padx=14, pady=2)

        self.lbl_progress = tk.Label(parent, text='Ready.',
                                      font=('Courier', 8), fg=FG_DIM, bg=BG_PANEL)
        self.lbl_progress.pack(anchor='w', padx=14)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=14, pady=8)

        # ── Results summary ──
        tk.Label(parent, text='RESULTS', font=('Courier', 9, 'bold'),
                 fg=ACCENT, bg=BG_PANEL).pack(anchor='w', padx=14)

        self.lbl_results = tk.Label(parent, text='—',
                                     font=('Courier', 9), fg=FG_DIM,
                                     bg=BG_PANEL, justify='left', wraplength=260)
        self.lbl_results.pack(anchor='w', padx=14, pady=4)

    def _build_notebook(self, parent):
        nb_style = ttk.Style()
        nb_style.configure('Dark.TNotebook', background=BG_DARK, borderwidth=0)
        nb_style.configure('Dark.TNotebook.Tab',
                            background=BG_MID, foreground=FG_DIM,
                            font=('Courier', 9, 'bold'), padding=[12, 5])
        nb_style.map('Dark.TNotebook.Tab',
                     background=[('selected', BG_PANEL)],
                     foreground=[('selected', ACCENT)])

        self.nb = ttk.Notebook(parent, style='Dark.TNotebook')
        self.nb.pack(fill='both', expand=True)

        # Tab 1 – Data Explorer
        tab_data = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(tab_data, text=' 🔭 DATA EXPLORER ')
        self._build_data_tab(tab_data)

        # Tab 2 – Training Monitor
        tab_train = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(tab_train, text=' 📈 TRAINING MONITOR ')
        self._build_train_tab(tab_train)

        # Tab 3 – Model Comparison
        tab_compare = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(tab_compare, text=' 🏆 MODEL COMPARISON ')
        self._build_compare_tab(tab_compare)

        # Tab 4 – Predictions
        tab_pred = tk.Frame(self.nb, bg=BG_DARK)
        self.nb.add(tab_pred, text=' 🎯 PREDICT ')
        self._build_predict_tab(tab_pred)

    # ── Tab: Data Explorer ─────────────────────────────────────────
    def _build_data_tab(self, parent):
        self.fig_data = Figure(figsize=(10, 5.5), facecolor=BG_DARK)
        self.fig_data.patch.set_facecolor(BG_DARK)
        canvas = FigureCanvasTkAgg(self.fig_data, master=parent)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas_data = canvas

    def _draw_data_tab(self):
        if self.df is None:
            return
        df = self.df
        fig = self.fig_data
        fig.clear()
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38,
                               top=0.92, bottom=0.1, left=0.07, right=0.97)

        # Class distribution
        ax1 = fig.add_subplot(gs[0, 0])
        counts = df['Type'].value_counts().sort_index()
        ax1.bar(range(6), counts.values,
                color=[STAR_COLORS[i] for i in range(6)], edgecolor=BORDER, lw=0.8)
        ax1.set_xticks(range(6))
        ax1.set_xticklabels([TYPE_LABELS[i] for i in range(6)],
                            rotation=30, ha='right', fontsize=7)
        ax1.set_title('Class Distribution', fontsize=10)

        # Temperature distribution
        ax2 = fig.add_subplot(gs[0, 1])
        for t in range(6):
            ax2.hist(df[df['Type'] == t]['Temperature'], bins=15, alpha=0.6,
                     color=STAR_COLORS[t], label=TYPE_LABELS[t], edgecolor='none')
        ax2.set_title('Temperature by Type', fontsize=10)
        ax2.set_xlabel('Temperature (K)', fontsize=8)
        ax2.legend(fontsize=6, facecolor=BG_MID, edgecolor=BORDER)

        # HR Diagram
        ax3 = fig.add_subplot(gs[0, 2])
        for t in range(6):
            sub = df[df['Type'] == t]
            ax3.scatter(sub['Temperature'], sub['L'], s=40, alpha=0.8,
                        color=STAR_COLORS[t], label=TYPE_LABELS[t],
                        edgecolors=BORDER, lw=0.3)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.invert_xaxis()
        ax3.set_title('HR Diagram', fontsize=10)
        ax3.set_xlabel('Temperature (K)', fontsize=8)
        ax3.set_ylabel('Luminosity (L☉)', fontsize=8)
        ax3.legend(fontsize=6, facecolor=BG_MID, edgecolor=BORDER)

        # Radius vs Luminosity
        ax4 = fig.add_subplot(gs[1, 0])
        for t in range(6):
            sub = df[df['Type'] == t]
            ax4.scatter(sub['R'], sub['L'], s=40, alpha=0.8,
                        color=STAR_COLORS[t], label=TYPE_LABELS[t],
                        edgecolors=BORDER, lw=0.3)
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_title('Radius vs Luminosity', fontsize=10)
        ax4.set_xlabel('Radius (R☉)', fontsize=8)
        ax4.set_ylabel('Luminosity (L☉)', fontsize=8)

        # Abs Magnitude distribution
        ax5 = fig.add_subplot(gs[1, 1])
        for t in range(6):
            ax5.hist(df[df['Type'] == t]['A_M'], bins=15, alpha=0.6,
                     color=STAR_COLORS[t], edgecolor='none')
        ax5.set_title('Absolute Magnitude Dist.', fontsize=10)
        ax5.set_xlabel('Abs. Magnitude', fontsize=8)

        # Spectral class breakdown
        ax6 = fig.add_subplot(gs[1, 2])
        spec_counts = df['Spectral_Class'].value_counts()
        ax6.bar(spec_counts.index, spec_counts.values,
                color=PALETTE[:len(spec_counts)], edgecolor=BORDER, lw=0.8)
        ax6.set_title('Spectral Class Counts', fontsize=10)
        ax6.set_xlabel('Spectral Class', fontsize=8)

        fig.suptitle('Dataset Overview', fontsize=12, color=FG_BRIGHT, y=0.98)
        self.canvas_data.draw()

    # ── Tab: Training Monitor ──────────────────────────────────────
    def _build_train_tab(self, parent):
        self.fig_train = Figure(figsize=(10, 5.5), facecolor=BG_DARK)
        canvas = FigureCanvasTkAgg(self.fig_train, master=parent)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas_train = canvas
        self._init_train_plot()

    def _init_train_plot(self):
        fig = self.fig_train
        fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.35,
                               top=0.91, bottom=0.1, left=0.08, right=0.97)

        self.ax_acc  = fig.add_subplot(gs[0, :])
        self.ax_loss = fig.add_subplot(gs[1, 0])
        self.ax_cm   = fig.add_subplot(gs[1, 1])

        for ax in [self.ax_acc, self.ax_loss]:
            ax.set_facecolor(BG_MID)
            ax.grid(True, alpha=0.3)

        self.ax_acc.set_title('Training & Validation Accuracy', fontsize=10)
        self.ax_acc.set_xlabel('Epoch', fontsize=8)
        self.ax_acc.set_ylabel('Accuracy', fontsize=8)

        self.ax_loss.set_title('Training & Validation Loss', fontsize=10)
        self.ax_loss.set_xlabel('Epoch', fontsize=8)
        self.ax_loss.set_ylabel('Loss', fontsize=8)

        self.ax_cm.set_title('Confusion Matrix', fontsize=10)

        self.line_tacc,  = self.ax_acc.plot([], [], color=PALETTE[0], lw=2, label='Train Acc')
        self.line_vacc,  = self.ax_acc.plot([], [], color=PALETTE[1], lw=2,
                                             linestyle='--', label='Val Acc')
        self.line_tloss, = self.ax_loss.plot([], [], color=PALETTE[2], lw=2, label='Train Loss')
        self.line_vloss, = self.ax_loss.plot([], [], color=PALETTE[3], lw=2,
                                              linestyle='--', label='Val Loss')

        self.ax_acc.legend(fontsize=8, facecolor=BG_MID, edgecolor=BORDER)
        self.ax_loss.legend(fontsize=8, facecolor=BG_MID, edgecolor=BORDER)

        fig.suptitle('Live Training Monitor', fontsize=12, color=FG_BRIGHT, y=0.97)
        self.canvas_train.draw()

    def _update_train_plot(self):
        """Redraw live training curves (called from main thread)."""
        epochs = list(range(1, len(self.train_acc_h) + 1))

        for line, data in [(self.line_tacc, self.train_acc_h),
                           (self.line_vacc, self.val_acc_h),
                           (self.line_tloss, self.train_loss_h),
                           (self.line_vloss, self.val_loss_h)]:
            line.set_xdata(epochs)
            line.set_ydata(data)

        for ax in [self.ax_acc, self.ax_loss]:
            ax.relim()
            ax.autoscale_view()

        self.canvas_train.draw_idle()

    def _draw_confusion_matrix(self, y_true, y_pred):
        import seaborn as sns
        ax = self.ax_cm
        ax.clear()
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[TYPE_LABELS[i] for i in range(6)],
                    yticklabels=[TYPE_LABELS[i] for i in range(6)],
                    ax=ax, linewidths=0.5, linecolor=BG_DARK,
                    cbar_kws={'shrink': 0.7})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=6.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6.5)
        ax.set_title('Confusion Matrix', fontsize=10)
        self.canvas_train.draw_idle()

    # ── Tab: Model Comparison ──────────────────────────────────────
    def _build_compare_tab(self, parent):
        self.fig_compare = Figure(figsize=(10, 5.5), facecolor=BG_DARK)
        canvas = FigureCanvasTkAgg(self.fig_compare, master=parent)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas_compare = canvas
        self._draw_compare_placeholder()

    def _draw_compare_placeholder(self):
        fig = self.fig_compare
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Train models to see comparison',
                ha='center', va='center', fontsize=14,
                color=FG_DIM, transform=ax.transAxes)
        ax.set_facecolor(BG_MID)
        self.canvas_compare.draw()

    def _draw_compare_tab(self):
        results = self.sk_results
        if not results:
            return
        import seaborn as sns
        fig = self.fig_compare
        fig.clear()
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.38,
                               top=0.91, bottom=0.1, left=0.08, right=0.97)

        model_names = list(results.keys())
        test_accs   = [results[m]['test_acc'] for m in model_names]

        # Add NN result if available
        all_names = model_names.copy()
        all_accs  = test_accs.copy()
        if self.nn_acc is not None:
            all_names.append('Neural Net')
            all_accs.append(self.nn_acc)

        short = [n.replace(' ', '\n') for n in all_names]
        colors = PALETTE[:len(all_names)]

        # Test accuracy bar
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(short, all_accs, color=colors, edgecolor=BORDER, lw=0.8)
        ax1.set_ylim(min(all_accs) * 0.98, 1.01)
        ax1.set_title('Test Accuracy', fontsize=10)
        ax1.set_ylabel('Accuracy', fontsize=8)
        for bar, acc in zip(bars, all_accs):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.001,
                     f'{acc:.3f}', ha='center', va='bottom',
                     fontsize=8, color=FG_BRIGHT, fontweight='bold')

        # CV scores with error bars
        ax2 = fig.add_subplot(gs[0, 1])
        sk_names = [m for m in model_names]
        cv_means = [results[m]['cv_mean'] for m in sk_names]
        cv_stds  = [results[m]['cv_std']  for m in sk_names]
        x2 = np.arange(len(sk_names))
        ax2.bar(x2, cv_means, color=PALETTE[:len(sk_names)], edgecolor=BORDER, lw=0.8)
        ax2.errorbar(x2, cv_means, yerr=[2 * s for s in cv_stds],
                     fmt='none', color=FG_BRIGHT, capsize=4, lw=1.5)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([n.split(' ')[0] + '\n' + ' '.join(n.split(' ')[1:])
                              for n in sk_names], fontsize=8)
        ax2.set_ylim(min(cv_means) * 0.97, 1.01)
        ax2.set_title('5-Fold CV Accuracy (±2σ)', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=8)

        # Feature importance (RF)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'Random Forest' in results:
            rf = results['Random Forest']['model']
            feat_labels = ['Temperature', 'Luminosity', 'Radius',
                           'Abs Mag', 'Color', 'Spec Class']
            imps = rf.feature_importances_
            idx  = np.argsort(imps)
            ax3.barh(np.array(feat_labels)[idx], imps[idx],
                     color=[PALETTE[i % len(PALETTE)] for i in idx],
                     edgecolor=BORDER, lw=0.5)
            ax3.set_title('Feature Importance (RF)', fontsize=10)
            ax3.set_xlabel('Importance', fontsize=8)

        # Per-class F1
        ax4 = fig.add_subplot(gs[1, 1])
        width = 0.25
        x4    = np.arange(6)
        for i, name in enumerate(sk_names[:3]):
            rep = classification_report(
                self.y_test, results[name]['model'].predict(self.X_test),
                output_dict=True)
            f1s = [rep[str(c)]['f1-score'] for c in range(6)]
            ax4.bar(x4 + i * width, f1s, width, label=name.split(' ')[0],
                    color=PALETTE[i], edgecolor=BORDER, lw=0.5, alpha=0.9)
        ax4.set_xticks(x4 + width)
        ax4.set_xticklabels([TYPE_LABELS[i] for i in range(6)],
                            rotation=25, ha='right', fontsize=7)
        ax4.set_ylim(0, 1.12)
        ax4.set_title('Per-Class F1 Score', fontsize=10)
        ax4.legend(fontsize=7, facecolor=BG_MID, edgecolor=BORDER)
        ax4.axhline(1.0, color=BORDER, linestyle='--', lw=0.8)

        fig.suptitle('Model Comparison Dashboard', fontsize=12, color=FG_BRIGHT, y=0.97)
        self.canvas_compare.draw()

    # ── Tab: Predict ───────────────────────────────────────────────
    def _build_predict_tab(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL)
        frame.pack(fill='both', expand=True, padx=20, pady=20)

        tk.Label(frame, text='MANUAL STAR PREDICTION',
                 font=('Courier', 13, 'bold'), fg=ACCENT, bg=BG_PANEL).pack(pady=(10, 4))
        tk.Label(frame, text='Enter star parameters to classify using the trained model.',
                 font=('Courier', 9), fg=FG_DIM, bg=BG_PANEL).pack(pady=(0, 14))

        grid_f = tk.Frame(frame, bg=BG_PANEL)
        grid_f.pack()

        self.pred_vars = {}
        params = [
            ('Temperature (K)', 'temp',   '3500'),
            ('Luminosity (L☉)', 'lum',    '0.001'),
            ('Radius (R☉)',     'radius', '0.15'),
            ('Abs. Magnitude',  'amag',   '16.0'),
        ]
        for row_i, (label, key, default) in enumerate(params):
            tk.Label(grid_f, text=label, font=('Courier', 10), fg=FG_DIM,
                     bg=BG_PANEL, width=18, anchor='e').grid(
                row=row_i, column=0, padx=8, pady=5)
            v = tk.StringVar(value=default)
            self.pred_vars[key] = v
            tk.Entry(grid_f, textvariable=v, font=('Courier', 10),
                     fg=FG_BRIGHT, bg=BG_MID, insertbackground=FG_BRIGHT,
                     relief='flat', width=14).grid(row=row_i, column=1, padx=8)

        # Color & spectral class dropdowns
        color_opts = ['Red', 'Blue', 'White', 'Yellow', 'Orange']
        spec_opts  = ['M', 'K', 'G', 'F', 'A', 'B', 'O']

        tk.Label(grid_f, text='Color', font=('Courier', 10), fg=FG_DIM,
                 bg=BG_PANEL, width=18, anchor='e').grid(row=4, column=0, padx=8, pady=5)
        self.pred_color = tk.StringVar(value='Red')
        ttk.Combobox(grid_f, textvariable=self.pred_color, values=color_opts,
                     width=12, state='readonly').grid(row=4, column=1, padx=8)

        tk.Label(grid_f, text='Spectral Class', font=('Courier', 10), fg=FG_DIM,
                 bg=BG_PANEL, width=18, anchor='e').grid(row=5, column=0, padx=8, pady=5)
        self.pred_spec = tk.StringVar(value='M')
        ttk.Combobox(grid_f, textvariable=self.pred_spec, values=spec_opts,
                     width=12, state='readonly').grid(row=5, column=1, padx=8)

        tk.Button(frame, text='⚡  CLASSIFY',
                  font=('Courier', 11, 'bold'), fg='#0D1117', bg=ACCENT,
                  relief='flat', cursor='hand2', activebackground='#8078FF',
                  command=self._on_predict_click, pady=6, padx=20).pack(pady=16)

        self.lbl_pred_result = tk.Label(frame, text='',
                                         font=('Courier', 16, 'bold'),
                                         fg=ACCENT, bg=BG_PANEL)
        self.lbl_pred_result.pack(pady=4)

        self.lbl_pred_detail = tk.Label(frame, text='',
                                         font=('Courier', 10),
                                         fg=FG_DIM, bg=BG_PANEL, justify='center')
        self.lbl_pred_detail.pack()

    # ── Data Loading ───────────────────────────────────────────────
    def _load_data(self):
        try:
            (self.df, self.X_scaled,
             self.X_train, self.X_test,
             self.y_train, self.y_test,
             self.features) = load_and_preprocess(self.CSV_PATH)

            n = len(self.df)
            self.lbl_dataset.config(
                text=f'{n} samples · 6 features · 6 classes\n'
                     f'Train: {len(self.X_train)}  Test: {len(self.X_test)}',
                fg=FG_BRIGHT)

            self.after(100, self._draw_data_tab)
        except FileNotFoundError:
            messagebox.showerror(
                'File Not Found',
                f'Could not find {self.CSV_PATH}.\n\n'
                'Please place Stars.csv in the same directory as this script.')
            self.lbl_dataset.config(text='Error: Stars.csv not found.', fg='#FF6584')

    # ── Training Logic ─────────────────────────────────────────────
    def _on_train_click(self):
        if self.training or self.df is None:
            return
        self.training = True
        self.btn_train.config(state='disabled', text='Training…')
        self.train_acc_h.clear()
        self.val_acc_h.clear()
        self.train_loss_h.clear()
        self.val_loss_h.clear()
        self._init_train_plot()
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        try:
            selected = self.model_var.get()
            train_all = self.var_train_all.get()

            sk_models_map = {
                'Random Forest'    : RandomForestClassifier(n_estimators=200, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=200,
                                                                 learning_rate=0.1,
                                                                 random_state=42),
                'SVM (RBF)'        : SVC(kernel='rbf', C=10, gamma='scale',
                                         random_state=42, probability=True),
            }

            models_to_run = list(sk_models_map.keys()) if train_all else (
                [selected] if selected in sk_models_map else [])

            total_sk = len(models_to_run)
            nn_needed = train_all or selected == 'Neural Network (TF)'

            # ── Scikit-learn models ──
            for i, name in enumerate(models_to_run):
                self._set_status(f'Training {name}…', (i / max(total_sk + nn_needed, 1)) * 100)
                m = sk_models_map[name]
                m.fit(self.X_train, self.y_train)
                cv  = cross_val_score(m, self.X_scaled, np.concatenate([self.y_train, self.y_test]),
                                      cv=5, scoring='accuracy')
                acc = accuracy_score(self.y_test, m.predict(self.X_test))
                self.sk_results[name] = {
                    'model': m, 'test_acc': acc,
                    'cv_mean': cv.mean(), 'cv_std': cv.std(),
                }

                # Show confusion matrix for last SK model
                self.after(0, lambda m=m: self._draw_confusion_matrix(
                    self.y_test, m.predict(self.X_test)))

            # ── Neural Network ──
            if nn_needed:
                idx = len(models_to_run)
                self._set_status('Building Neural Network…', (idx / (total_sk + 1)) * 100)

                epochs    = int(self.var_epochs.get())
                batch     = int(self.var_batch.get())
                dropout   = float(self.var_drop.get())
                units     = (int(self.var_l1.get()),
                             int(self.var_l2.get()),
                             int(self.var_l3.get()))

                tf.random.set_seed(42)
                self.nn_model = build_nn(self.X_train.shape[1], 6, units, dropout)
                self.nn_model.compile(
                    optimizer=keras.optimizers.Adam(1e-3),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

                cb = [
                    LiveCallback(self),
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                                   restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10,
                                                       min_lr=1e-5),
                ]
                self.nn_model.fit(
                    self.X_train, self.y_train,
                    validation_split=0.15,
                    epochs=epochs, batch_size=batch,
                    callbacks=cb, verbose=0)

                self.nn_acc = self.nn_model.evaluate(self.X_test, self.y_test, verbose=0)[1]
                nn_preds    = np.argmax(self.nn_model.predict(self.X_test, verbose=0), axis=1)
                self.after(0, lambda: self._draw_confusion_matrix(self.y_test, nn_preds))

            self.after(0, self._training_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror('Training Error', str(e)))
            self.after(0, self._training_done)

    def on_epoch_end(self, epoch, logs):
        """Called from Keras callback — safe to schedule UI updates."""
        self.train_acc_h.append(logs.get('accuracy', 0))
        self.val_acc_h.append(logs.get('val_accuracy', 0))
        self.train_loss_h.append(logs.get('loss', 0))
        self.val_loss_h.append(logs.get('val_loss', 0))

        epochs    = int(self.var_epochs.get())
        pct = min((epoch + 1) / epochs * 100, 100)
        status = (f'Epoch {epoch+1}/{epochs}  '
                  f'acc={logs.get("accuracy", 0):.3f}  '
                  f'val_acc={logs.get("val_accuracy", 0):.3f}')
        self.after(0, lambda: self._set_status(status, pct))
        self.after(0, self._update_train_plot)

    def _training_done(self):
        self.training = False
        self.btn_train.config(state='normal', text='▶  START TRAINING')
        self._set_status('Training complete!', 100)

        # Build result summary text
        lines = []
        for name, r in self.sk_results.items():
            lines.append(f'{name[:14]:<14}  {r["test_acc"]:.4f}')
        if self.nn_acc is not None:
            lines.append(f'{"Neural Net":<14}  {self.nn_acc:.4f}')
        self.lbl_results.config(text='\n'.join(lines) if lines else '—', fg=FG_BRIGHT)

        self._draw_compare_tab()
        # Switch to compare tab
        self.nb.select(2)

    def _set_status(self, msg, pct):
        self.lbl_progress.config(text=msg)
        self.progress_var.set(pct)

    # ── Predict Logic ──────────────────────────────────────────────
    def _on_predict_click(self):
        if not self.sk_results and self.nn_model is None:
            messagebox.showinfo('No Model', 'Please train a model first.')
            return
        try:
            temp   = float(self.pred_vars['temp'].get())
            lum    = float(self.pred_vars['lum'].get())
            radius = float(self.pred_vars['radius'].get())
            amag   = float(self.pred_vars['amag'].get())

            # Encode color & spectral class using a simple mapping
            color_map = {'red': 0, 'blue': 1, 'white': 2, 'yellow': 3, 'orange': 4}
            spec_map  = {'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6}
            color_enc = color_map.get(self.pred_color.get().lower(), 0)
            spec_enc  = spec_map.get(self.pred_spec.get().upper(), 0)

            x = np.array([[np.log1p(temp), np.log1p(lum), np.log1p(radius),
                           amag, color_enc, spec_enc]], dtype=float)

            # Re-fit scaler for prediction (use raw features first 4 raw, rest encoded)
            # Use nn_model if available, else first SK model
            if self.nn_model is not None:
                (_, X_sc, _, _, _, _, _) = load_and_preprocess(self.CSV_PATH)
                sc2 = StandardScaler()
                sc2.fit(X_sc)
                x_scaled = sc2.transform(x)
                probs = self.nn_model.predict(x_scaled, verbose=0)[0]
                pred  = int(np.argmax(probs))
                conf  = float(probs[pred])
                model_used = 'Neural Network'
            else:
                name  = list(self.sk_results.keys())[0]
                (_, X_sc, _, _, _, _, _) = load_and_preprocess(self.CSV_PATH)
                sc2 = StandardScaler()
                sc2.fit(X_sc)
                x_scaled = sc2.transform(x)
                m     = self.sk_results[name]['model']
                pred  = int(m.predict(x_scaled)[0])
                probs = m.predict_proba(x_scaled)[0] if hasattr(m, 'predict_proba') else None
                conf  = float(probs[pred]) if probs is not None else 1.0
                model_used = name

            star_type = TYPE_LABELS[pred]
            self.lbl_pred_result.config(
                text=f'⭐  {star_type}',
                fg=STAR_COLORS[pred])
            self.lbl_pred_detail.config(
                text=f'Confidence: {conf * 100:.1f}%   •   Model: {model_used}')
        except Exception as e:
            messagebox.showerror('Prediction Error', str(e))


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = StarClassifierApp()
    app.mainloop()