"""
Spelling Correction System — GUI (Tkinter)
CT052-3-M-NLP Assignment — Part B Demo
Students
- TP139069 - Muhammad Mahdi Hardi Effendi
- TP131138 - Alieu Bojang

Requirements fulfilled:
  1. Sorted dictionary list with search facility
  2. 500-character text editor
  3. Highlight misspelled words
  4. Click on misspelled word → suggest corrections with edit distances
  5. Non-word AND real-word error correction
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import re
import math
from collections import Counter, defaultdict

# =========================
# 1. NLP BACKEND  (same as the jupyter notebook)
# ========================

import nltk
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import brown
from nltk.probability import FreqDist
from nltk.lm import Laplace as LaplaceLM
from nltk.lm.preprocessing import padded_everygram_pipeline


def build_backend(status_callback=None):
    """Build all NLP models. Returns a dict of everything the GUI needs."""

    def update(msg):
        if status_callback:
            status_callback(msg)

    # --- Corpus ---
    update("Loading Brown corpus...")
    raw_words = brown.words()
    corpus_words = [w.lower() for w in raw_words if w.isalpha()]
    word_counts = FreqDist(corpus_words)
    VOCABULARY = set(word_counts.keys())
    TOTAL_WORDS = word_counts.N()
    VOCAB_SIZE = word_counts.B()

    # --- Word probability (Laplace) ---
    def word_probability(word):
        return (word_counts[word] + 1) / (TOTAL_WORDS + VOCAB_SIZE)

    # --- Bigram Language Model ---
    update("Training bigram language model...")
    tokenized_sents = []
    for sent in brown.sents():
        clean = [w.lower() for w in sent if w.isalpha()]
        if clean:
            tokenized_sents.append(clean)

    train_data, vocab = padded_everygram_pipeline(2, tokenized_sents)
    bigram_model = LaplaceLM(order=2)
    bigram_model.fit(train_data, vocab)

    def bigram_probability(word, prev_word):
        return bigram_model.score(word, [prev_word])

    def sentence_log_probability(words):
        log_prob = 0.0
        prev = '<s>'
        for w in words:
            log_prob += math.log(bigram_probability(w, prev))
            prev = w
        return log_prob

    # --- Edit distance ---
    def min_edit_distance(source, target, sub_cost=1):
        n, m = len(source), len(target)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1): dp[i][0] = i
        for j in range(m+1): dp[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = 0 if source[i-1] == target[j-1] else sub_cost
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
                if i>1 and j>1 and source[i-1]==target[j-2] and source[i-2]==target[j-1]:
                    dp[i][j] = min(dp[i][j], dp[i-2][j-2]+1)
        return dp[n][m]

    # --- Candidate generation ---
    def edits_one(word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
        deletes    = [L+R[1:]             for L,R in splits if R]
        transposes = [L+R[1]+R[0]+R[2:]   for L,R in splits if len(R)>1]
        replaces   = [L+c+R[1:]           for L,R in splits if R for c in letters]
        inserts    = [L+c+R               for L,R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits_two(word):
        return {e2 for e1 in edits_one(word) for e2 in edits_one(e1) if e2 in VOCABULARY}

    # --- Confusion matrices ---
    update("Building confusion matrices from spell-errors.txt...")

    def identify_edit(source, target):
        ls, lt = len(source), len(target)
        if ls == lt - 1:
            for i in range(lt):
                if source == target[:i] + target[i+1:]:
                    prev = target[i-1] if i > 0 else '#'
                    return ('del', prev, target[i])
        if ls == lt + 1:
            for i in range(ls):
                if target == source[:i] + source[i+1:]:
                    prev = source[i-1] if i > 0 else '#'
                    return ('ins', prev, source[i])
        if ls == lt:
            diffs = [(i, source[i], target[i]) for i in range(ls) if source[i] != target[i]]
            if len(diffs) == 1:
                return ('sub', diffs[0][1], diffs[0][2])
            if len(diffs) == 2:
                i, j = diffs[0][0], diffs[1][0]
                if j == i+1 and source[i] == target[j] and source[j] == target[i]:
                    return ('trans', source[i], source[j])
        return None

    char_bigram_counts = Counter()
    char_unigram_counts = Counter()
    for w in corpus_words:
        for ch in w:
            char_unigram_counts[ch] += 1
        for i in range(len(w)-1):
            char_bigram_counts[w[i]+w[i+1]] += 1

    del_matrix   = defaultdict(lambda: defaultdict(lambda: 1))
    ins_matrix   = defaultdict(lambda: defaultdict(lambda: 1))
    sub_matrix   = defaultdict(lambda: defaultdict(lambda: 1))
    trans_matrix = defaultdict(lambda: defaultdict(lambda: 1))

    import os
    spell_errors_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spell-errors.txt')
    if not os.path.exists(spell_errors_path):
        spell_errors_path = 'spell-errors.txt'

    with open(spell_errors_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            correct, typos_str = line.split(':', 1)
            correct = correct.strip().lower()
            for typo in typos_str.split(','):
                typo = typo.strip()
                if '*' in typo:
                    typo = typo.split('*')[0].strip()
                if typo and typo != correct:
                    edit = identify_edit(typo, correct)
                    if edit:
                        op, c1, c2 = edit
                        if   op == 'del':   del_matrix[c1][c2]   += 10
                        elif op == 'ins':   ins_matrix[c1][c2]   += 10
                        elif op == 'sub':   sub_matrix[c1][c2]   += 10
                        elif op == 'trans': trans_matrix[c1][c2] += 10

    def channel_probability(typo, candidate):
        edit = identify_edit(typo, candidate)
        if edit is None:
            dist = min_edit_distance(typo, candidate)
            return math.exp(-2 * dist)
        op, c1, c2 = edit
        if op == 'del':
            return del_matrix[c1][c2] / (char_bigram_counts.get(c1+c2, 1) + 26)
        elif op == 'ins':
            return ins_matrix[c1][c2] / (char_unigram_counts.get(c1, 1) + 26)
        elif op == 'sub':
            return sub_matrix[c1][c2] / (char_unigram_counts.get(c2, 1) + 26)
        elif op == 'trans':
            return trans_matrix[c1][c2] / (char_bigram_counts.get(c1+c2, 1) + 26)
        return 1e-6

    # --- Get candidates with scores ---
    def get_candidates(word):
        if word in VOCABULARY:
            return {word}
        c1 = edits_one(word) & VOCABULARY
        if c1:
            return c1
        c2 = edits_two(word)
        return c2 if c2 else {word}

    def correct_nonword(word):
        word = word.lower()
        if word in VOCABULARY:
            return word, []
        candidates = get_candidates(word)
        scored = []
        for c in candidates:
            p_xw  = channel_probability(word, c)
            p_w   = word_probability(c)
            score = p_xw * p_w
            scored.append((c, p_xw, p_w, score))
        scored.sort(key=lambda x: x[3], reverse=True)
        return (scored[0][0] if scored else word), scored[:10]

    def correct_realword(sentence):
        words = sentence.lower().split()
        original_score = sentence_log_probability(words)
        best_sentence = list(words)
        best_score = original_score
        corrections = []
        for i, word in enumerate(words):
            if word not in VOCABULARY:
                continue
            candidates = (edits_one(word) & VOCABULARY) | {word}
            for candidate in candidates:
                if candidate == word:
                    continue
                test_words = list(words)
                test_words[i] = candidate
                test_score = sentence_log_probability(test_words)
                if test_score > best_score:
                    best_sentence = list(test_words)
                    best_score = test_score
                    corrections = [(word, candidate, i)]
        return ' '.join(best_sentence), corrections

    def spell_correct(sentence):
        words = sentence.lower().split()
        all_corrections = []
        corrected_words = []
        for word in words:
            if word not in VOCABULARY:
                fixed, scores = correct_nonword(word)
                if fixed != word:
                    all_corrections.append({
                        'type': 'non-word', 'original': word,
                        'corrected': fixed,
                        'edit_distance': min_edit_distance(word, fixed)
                    })
                corrected_words.append(fixed)
            else:
                corrected_words.append(word)
        after_pass1 = ' '.join(corrected_words)
        final, rw_corr = correct_realword(after_pass1)
        for orig, fix, pos in rw_corr:
            all_corrections.append({
                'type': 'real-word', 'original': orig,
                'corrected': fix,
                'edit_distance': min_edit_distance(orig, fix)
            })
        return final, all_corrections

    update("Ready!")
    return {
        'VOCABULARY': VOCABULARY,
        'word_counts': word_counts,
        'min_edit_distance': min_edit_distance,
        'correct_nonword': correct_nonword,
        'correct_realword': correct_realword,
        'spell_correct': spell_correct,
        'get_candidates': get_candidates,
        'channel_probability': channel_probability,
        'word_probability': word_probability,
        'edits_one': edits_one,
    }


# =============================================================
# 2. GUI  — The visual interface that the user interacts with.
#    It uses Tkinter, Python's built-in GUI library.
#    Some function that are used:
#      - tk.Frame   = a <div> container
#      - tk.Label   = a <span> or <p> for text
#      - tk.Text    = a <textarea> for multi-line input
#      - tk.Listbox = a <select> dropdown / scrollable list
#      - ttk.Button = a <button>
#      - ttk.Treeview = an HTML <table>
# =============================================================

class SpellCheckerGUI:
    MAX_CHARS = 500  # Assignment requires max 500 characters in the text editor

    def __init__(self, root):
        """
        Constructor — runs once when the app starts.
        'root' is the main application window created by tk.Tk() at the bottom.
        """
        self.root = root
        self.root.title("Spelling Correction System — CT052-3-M-NLP")  # Window title bar text
        self.root.geometry("1100x720")   # Default window size: 1100px wide × 720px tall
        self.root.minsize(900, 600)      # User can't resize smaller than this

        self.backend = None           # Will hold ALL the NLP functions after loading finishes
        self.misspelled_ranges = {}   # Tracks highlighted words: {'err_0': 'teh', 'err_1': 'mney', ...}
        self._selected_tag = None     # Which highlighted word the user last clicked on

        self._build_ui()       # Step 1: Create all the visual elements (buttons, text boxes, etc.)
        self._start_loading()  # Step 2: Start loading NLP models in the background

    # ----------------------------------------------------
    # UI CONSTRUCTION — builds every visual element on screen
    # ----------------------------------------------------

    def _build_ui(self):
        """
        Builds the entire GUI layout. Called once during __init__.

        Layout structure:
        ┌──────────────────────────────────────────────┐
        │  [Title]                        [Status msg] │  ← top bar
        ├──────────────┬───────────────────────────────┤
        │  Dictionary  │  Text Editor (500 chars max)  │
        │  (sorted     │  [Check] [Auto-Correct] [Clr] │
        │   word list  │                               │
        │   + search)  │  Suggestions Table            │
        │              │  (candidate, dist, scores)    │
        └──────────────┴───────────────────────────────┘
        """

        # --- Configure custom font styles ---
        # ttk.Style() lets us define reusable styles for widgets (like CSS classes)
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'))  # Big bold title
        style.configure('Status.TLabel', font=('Segoe UI', 10))         # Smaller status text

        # -- TOP BAR -------------------------------
        # A horizontal strip at the top with the title and status message
        top = ttk.Frame(self.root, padding=8)  # Frame = invisible container to group widgets
        top.pack(fill=tk.X)                    # pack() places it in the window; fill=X stretches horizontally

        # Title label on the left side
        ttk.Label(top, text="Spelling Correction System", style='Title.TLabel').pack(side=tk.LEFT)

        # Status message on the right side — changes dynamically
        # StringVar is a special Tkinter variable: when its value changes,
        # any Label connected to it updates automatically (like reactive state in React)
        self.status_var = tk.StringVar(value="Loading models…")
        ttk.Label(top, textvariable=self.status_var, style='Status.TLabel').pack(side=tk.RIGHT)

        # -- MAIN PANED AREA -------------------------------
        # PanedWindow splits the screen into resizable left/right panels
        # The user can drag the divider between them
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        # fill=BOTH = stretch in both directions
        # expand=True = take up all available space
        # padx/pady = outer margin (padding around the widget)

        # -- LEFT PANEL: DICTIONARY -------------------------------
        # LabelFrame = a Frame with a visible border and a title label
        left = ttk.LabelFrame(pane, text="Dictionary (Brown Corpus)", padding=6)
        pane.add(left, weight=1)  # weight=1 means this panel gets 1 share of space

        # Search bar: a text field that filters the dictionary as you type
        sf = ttk.Frame(left)                           # Container for the search row
        sf.pack(fill=tk.X, pady=(0, 4))                # Stretch horizontally, 4px bottom margin
        ttk.Label(sf, text="Search:").pack(side=tk.LEFT)  # "Search:" label on the left

        self.dict_search_var = tk.StringVar()          # Holds whatever the user types in the search box
        # trace_add('write', callback) means: every time this variable changes (user types),
        # call self._filter_dict. This is how we get "live search" / "search as you type"
        self.dict_search_var.trace_add('write', self._filter_dict)
        # The actual text input field, connected to the StringVar
        ttk.Entry(sf, textvariable=self.dict_search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # Dictionary word list — a scrollable list of all vocabulary words
        self.dict_list = tk.Listbox(left, font=('Consolas', 10))  # Monospace font for alignment
        # Scrollbar: connects to the Listbox so scrolling works both ways
        dict_scroll = ttk.Scrollbar(left, command=self.dict_list.yview)      # Scrollbar controls Listbox
        self.dict_list.configure(yscrollcommand=dict_scroll.set)             # Listbox controls Scrollbar
        dict_scroll.pack(side=tk.RIGHT, fill=tk.Y)   # Scrollbar on the right edge, full height
        self.dict_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Listbox fills remaining space

        # Word count label below the list (e.g., "40,234 words")
        self.dict_count_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.dict_count_var).pack(anchor=tk.W, pady=(4, 0))

        # -- RIGHT PANEL ----------------------------------------
        right = ttk.Frame(pane, padding=6)
        pane.add(right, weight=2)  # weight=2 = gets TWICE the space of the left panel

        # -- TEXT EDITOR -----------------------------------------
        # Where the user types or pastes their text to check
        editor_frame = ttk.LabelFrame(right, text="Text Editor (max 500 characters)", padding=4)
        editor_frame.pack(fill=tk.BOTH, expand=True)

        # tk.Text = multi-line text editor (like <textarea> in HTML)
        # wrap=WORD means lines break at word boundaries, not mid-word
        # height=8 means roughly 8 lines tall (but it stretches because of expand=True above)
        self.text = tk.Text(editor_frame, font=('Segoe UI', 12), wrap=tk.WORD, height=8)
        self.text.pack(fill=tk.BOTH, expand=True)

        # Bind events:
        # '<KeyRelease>' = after any key is pressed and released → count characters
        self.text.bind('<KeyRelease>', self._on_key)
        # '<Button-1>' = left mouse click → check if user clicked on a highlighted word
        self.text.bind('<Button-1>', self._on_click)

        # Character counter label (e.g., "142 / 500 characters")
        self.char_count_var = tk.StringVar(value="0 / 500 characters")
        ttk.Label(editor_frame, textvariable=self.char_count_var).pack(anchor=tk.E)  # anchor=E = right-aligned

        # -- BUTTONS ROW ----------------------------------------
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=6)

        # "Check Spelling" button — calls _check_spelling when clicked
        # state=DISABLED means it's greyed out and unclickable (until models finish loading)
        self.check_btn = ttk.Button(btn_frame, text="Check Spelling",
                                     command=self._check_spelling, state=tk.DISABLED)
        self.check_btn.pack(side=tk.LEFT, padx=(0, 6))

        # "Auto-Correct All" button — runs the full two-pass correction pipeline
        self.correct_btn = ttk.Button(btn_frame, text="Auto-Correct All",
                                       command=self._auto_correct, state=tk.DISABLED)
        self.correct_btn.pack(side=tk.LEFT, padx=(0, 6))

        # "Clear" button — always enabled, clears everything
        ttk.Button(btn_frame, text="Clear", command=self._clear).pack(side=tk.LEFT)

        # -- SUGGESTIONS TABLE ----------------------------------------
        # Shows correction candidates when user clicks a highlighted word
        sug_frame = ttk.LabelFrame(right, text="Suggestions (click a highlighted word)", padding=4)
        sug_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        # Treeview = a table/grid widget with columns (like an HTML <table>)
        # Define the 5 columns
        cols = ('candidate', 'edit_dist', 'channel', 'prior', 'score')
        self.sug_tree = ttk.Treeview(sug_frame, columns=cols, show='headings', height=6)
        # show='headings' means only show column headers, not the tree expand/collapse icons
        # height=6 means show roughly 6 rows

        # Set up each column's header text and width
        self.sug_tree.heading('candidate',  text='Candidate')   # The suggested correction word
        self.sug_tree.heading('edit_dist',  text='Edit Dist')   # How many edits away from the typo
        self.sug_tree.heading('channel',    text='P(x|w)')      # Channel model probability
        self.sug_tree.heading('prior',      text='P(w)')        # Language model (word frequency)
        self.sug_tree.heading('score',      text='Score')       # Combined: P(x|w) × P(w)

        # Column widths and alignment
        self.sug_tree.column('candidate', width=120)
        self.sug_tree.column('edit_dist', width=70, anchor=tk.CENTER)   # Center-aligned
        self.sug_tree.column('channel',   width=100, anchor=tk.E)       # Right-aligned (for numbers)
        self.sug_tree.column('prior',     width=100, anchor=tk.E)
        self.sug_tree.column('score',     width=100, anchor=tk.E)
        self.sug_tree.pack(fill=tk.BOTH, expand=True)

        # '<Double-1>' = double left-click on a table row → apply that suggestion
        self.sug_tree.bind('<Double-1>', self._apply_suggestion)

        # Label below the table (e.g., 'Suggestions for "teh" — double-click to apply')
        self.sug_label_var = tk.StringVar(value="")
        ttk.Label(sug_frame, textvariable=self.sug_label_var).pack(anchor=tk.W, pady=(2, 0))

        # -- HIGHLIGHT STYLE ----------------------------------------
        # Configure how misspelled words look in the text editor:
        # pink background (#FFCCCC) + underline
        # "Tags" in Tkinter Text widget are like CSS classes — you apply them to ranges of text
        self.text.tag_configure('misspelled', background='#FFCCCC', underline=True)

    # ---------------------------------------------------------------
    # BACKGROUND LOADING — loads NLP models without freezing the GUI
    # ---------------------------------------------------------------

    def _start_loading(self):
        """
        Starts loading the NLP backend in a BACKGROUND THREAD.

        Why a thread? build_backend() takes ~10 seconds (loading corpus,
        training bigram model, building confusion matrices). If we ran it
        on the main thread, the entire window would freeze — no clicks,
        no typing, nothing — until it finishes. That's terrible UX.

        Threading lets the heavy work happen in the background while the
        GUI stays responsive and shows "Loading models…" in the status bar.
        """
        def load():
            # build_backend() returns a dict of all NLP functions
            # The status_callback updates the status bar as each step finishes:
            #   "Loading Brown corpus..." -> "Training bigram model..." -> "Ready!"
            #
            # root.after(0, func, args) is CRITICAL here:
            #   Tkinter is NOT thread-safe — you can't update widgets from a background thread.
            #   root.after(0, ...) schedules the update to run on the MAIN thread safely.
            self.backend = build_backend(
                status_callback=lambda m: self.root.after(0, self.status_var.set, m)
            )
            # Once loading is done, schedule _on_loaded to run on the main thread
            self.root.after(0, self._on_loaded)

        # daemon=True means this thread dies automatically when the main window closes
        # (otherwise closing the window would leave a zombie thread running)
        threading.Thread(target=load, daemon=True).start()

    def _on_loaded(self):
        """
        Called AFTER build_backend() finishes (on the main thread).
        Enables the buttons and fills the dictionary list.
        """
        # Update status bar to show vocabulary size
        self.status_var.set(f"Ready — {len(self.backend['VOCABULARY']):,} words in dictionary")

        # Enable the buttons — they were DISABLED during loading
        self.check_btn.config(state=tk.NORMAL)    # NORMAL = clickable
        self.correct_btn.config(state=tk.NORMAL)

        # Sort the vocabulary alphabetically and populate the left panel list
        self.sorted_vocab = sorted(self.backend['VOCABULARY'])
        self._populate_dict(self.sorted_vocab)

    def _populate_dict(self, words):
        """
        Fills the dictionary Listbox with the given list of words.
        Called on startup (all words) and on every search keystroke (filtered words).
        """
        self.dict_list.delete(0, tk.END)   # Clear the list first (0 to END = everything)
        for w in words:
            self.dict_list.insert(tk.END, w)  # Append each word to the bottom
        self.dict_count_var.set(f"{len(words):,} words")  # Update count label, e.g. "40,234 words"

    def _filter_dict(self, *_):
        """
        Called every time the user types in the dictionary search box.
        Filters the word list to only show words starting with the query.

        *_ catches the extra arguments Tkinter passes to trace callbacks
        (variable name, index, mode) — we don't need them, so we ignore them.

        Example: user types "mar" → shows ['march', 'margin', 'market', 'marriage', ...]
        """
        query = self.dict_search_var.get().lower().strip()
        if not self.backend:
            return  # Models not loaded yet, nothing to filter
        if not query:
            self._populate_dict(self.sorted_vocab)  # Empty search → show all words
        else:
            # List comprehension: keep only words that START WITH the query
            filtered = [w for w in self.sorted_vocab if w.startswith(query)]
            self._populate_dict(filtered)

    # ----------------------------------------------------
    # EDITOR HELPERS — character counting and clearing
    # ----------------------------------------------------

    def _on_key(self, event=None):
        """
        Called after EVERY keystroke in the text editor.
        Enforces the 500-character limit and updates the counter.
        """
        content = self.text.get('1.0', 'end-1c')  # Get all text from the editor
        # '1.0' = line 1, character 0 (the very beginning)
        # 'end-1c' = the end minus 1 character (Tkinter adds a trailing newline; we skip it)
        length = len(content)

        if length > self.MAX_CHARS:
            # If over 500 chars, DELETE everything after character 500
            self.text.delete(f'1.0+{self.MAX_CHARS}c', tk.END)
            # f'1.0+500c' means "start of text + 500 characters"
            length = self.MAX_CHARS

        # Update the counter label: "342 / 500 characters"
        self.char_count_var.set(f"{length} / {self.MAX_CHARS} characters")

    def _clear(self):
        """Called when user clicks the 'Clear' button. Resets everything."""
        self.text.delete('1.0', tk.END)   # Clear all text in the editor
        self._clear_highlights()           # Remove all pink highlights
        self.sug_tree.delete(*self.sug_tree.get_children())  # Clear all rows in the suggestions table
        # get_children() returns IDs of all rows; * unpacks them as arguments to delete()
        self.sug_label_var.set("")         # Clear the suggestion label
        self.char_count_var.set("0 / 500 characters")  # Reset the counter

    # ──────────────────────────────────────────────────────────
    # SPELL CHECKING — detection, highlighting, and suggestions
    # ──────────────────────────────────────────────────────────

    def _clear_highlights(self):
        """
        Removes all misspelled word highlights from the text editor.
        Called before re-checking (so old highlights don't linger).
        """
        # Delete each individual error tag (err_0, err_1, etc.)
        for tag in list(self.misspelled_ranges.keys()):
            self.text.tag_delete(tag)
        self.misspelled_ranges.clear()  # Empty the tracking dictionary

        # Also remove the shared 'misspelled' tag (the pink highlight) from all text
        self.text.tag_remove('misspelled', '1.0', tk.END)
        self._selected_tag = None  # No word is selected anymore

    def _check_spelling(self):
        """
        Main spell-check function. Called when user clicks "Check Spelling".

        How it works:
        1. Get all the text from the editor
        2. Find every word using regex
        3. Check each word against the vocabulary dictionary
        4. If a word is NOT in the dictionary → highlight it pink
        5. Each highlighted word gets a unique tag so we can track it
        """
        if not self.backend:
            return  # Safety check: models not loaded yet

        # Clear any previous highlights and suggestions
        self._clear_highlights()
        self.sug_tree.delete(*self.sug_tree.get_children())
        self.sug_label_var.set("")

        content = self.text.get('1.0', 'end-1c')   # Get all text
        VOCAB = self.backend['VOCABULARY']           # The set of 40K known words

        error_count = 0

        # re.finditer(r'[a-zA-Z]+', content) finds every sequence of letters in the text.
        # It returns match objects with:
        #   m.group() = the word itself (e.g., "teh")
        #   m.start() = character position where the word starts (e.g., 23)
        #   m.end()   = character position where it ends (e.g., 26)
        for m in re.finditer(r'[a-zA-Z]+', content):
            word = m.group().lower()

            if word not in VOCAB:
                # This word is misspelled! Highlight it.

                # Convert character positions to Tkinter's text index format:
                # '1.0+23c' means "line 1, character 0, plus 23 characters forward"
                start_idx = f'1.0+{m.start()}c'
                end_idx   = f'1.0+{m.end()}c'

                # Create a unique tag name for this specific error
                tag = f'err_{error_count}'   # e.g., 'err_0', 'err_1', 'err_2'

                # Apply the shared 'misspelled' tag (pink background + underline)
                self.text.tag_add('misspelled', start_idx, end_idx)

                # Also apply the unique tag (so we know WHICH word was clicked later)
                self.text.tag_add(tag, start_idx, end_idx)

                # Bind a click handler to this specific tag:
                # When the user clicks on THIS word, call _on_word_click with this tag
                # The lambda captures the current value of 'tag' via the t=tag default argument
                # (without t=tag, all lambdas would reference the LAST value of tag — a common Python gotcha)
                self.text.tag_bind(tag, '<Button-1>', lambda e, t=tag: self._on_word_click(t))

                # Remember which word this tag represents
                self.misspelled_ranges[tag] = word  # e.g., {'err_0': 'teh', 'err_1': 'mney'}
                error_count += 1

        # Update the status bar
        if error_count == 0:
            self.status_var.set("No spelling errors found!")
        else:
            self.status_var.set(f"Found {error_count} misspelled word(s) — click one for suggestions")

    def _on_click(self, event):
        """
        Called on ANY left-click in the text editor.
        Checks if the click landed on a highlighted (misspelled) word.

        event.x, event.y = pixel coordinates of where the mouse clicked.
        We convert that to a text position and check if it's inside any error tag.
        """
        # Convert pixel coordinates → text index (like '1.15' meaning line 1, char 15)
        idx = self.text.index(f'@{event.x},{event.y}')

        # Check each error tag to see if the click is inside its range
        for tag in self.misspelled_ranges:
            ranges = self.text.tag_ranges(tag)  # Returns (start_index, end_index) of the tag
            if ranges:
                start, end = ranges[0], ranges[1]
                # text.compare() compares two text positions
                # Is the click position >= start AND <= end of this tag?
                if self.text.compare(idx, '>=', start) and self.text.compare(idx, '<=', end):
                    self._on_word_click(tag)  # Yes! Show suggestions for this word
                    return  # Stop checking — we found the word

    def _on_word_click(self, tag):
        """
        Called when the user clicks on a specific highlighted (misspelled) word.
        Populates the suggestions table with correction candidates.

        tag: the unique tag name like 'err_0' that identifies which word was clicked.
        """
        if not self.backend:
            return

        word = self.misspelled_ranges.get(tag)  # Look up which word this tag represents
        if not word:
            return

        self._selected_tag = tag  # Remember which word is selected (for applying suggestions later)

        # Clear the previous suggestions from the table
        self.sug_tree.delete(*self.sug_tree.get_children())

        # Call the NLP backend to get correction candidates
        # correct_nonword() returns: (best_word, list_of_scored_candidates)
        # Each candidate is a tuple: (word, P(x|w), P(w), score)
        _, scored = self.backend['correct_nonword'](word)
        med = self.backend['min_edit_distance']  # Get the edit distance function

        # Update the label above the table
        self.sug_label_var.set(f"Suggestions for \"{word}\" — double-click to apply")

        # Insert each candidate as a row in the suggestions table
        for c, pxw, pw, score in scored:
            dist = med(word, c)   # Calculate edit distance between the typo and this candidate
            self.sug_tree.insert('', tk.END, values=(
                c,                  # Candidate word (e.g., "the")
                dist,               # Edit distance (e.g., 1)
                f'{pxw:.6f}',       # Channel probability P(x|w) formatted to 6 decimal places
                f'{pw:.2e}',        # Prior P(w) in scientific notation (e.g., 6.85e-02)
                f'{score:.2e}'      # Combined score in scientific notation
            ))

    def _apply_suggestion(self, event):
        """
        Called when the user DOUBLE-CLICKS a row in the suggestions table.
        Replaces the misspelled word in the text editor with the selected candidate.
        """
        # Get the currently selected row in the table
        sel = self.sug_tree.selection()  # Returns a tuple of selected item IDs
        if not sel or not self._selected_tag:
            return  # Nothing selected, or no word was clicked earlier

        # Get the candidate word from the first column of the selected row
        candidate = self.sug_tree.item(sel[0])['values'][0]

        tag = self._selected_tag

        # Find WHERE the misspelled word is in the text editor
        ranges = self.text.tag_ranges(tag)  # Returns (start_position, end_position)
        if ranges:
            start, end = ranges[0], ranges[1]

            # Replace the misspelled word with the candidate:
            self.text.delete(start, end)         # Delete the old word
            self.text.insert(start, candidate)   # Insert the new word at the same position

            # Clean up: remove this tag since the word is now fixed
            self.text.tag_delete(tag)
            del self.misspelled_ranges[tag]

        # Clear the suggestions table
        self.sug_tree.delete(*self.sug_tree.get_children())
        self.sug_label_var.set(f"Applied: {candidate}")
        self._selected_tag = None

        # Re-run spell check to update all highlights
        # (because replacing a word shifts character positions, so old tags would be wrong)
        self._check_spelling()

    # ----------------------------------------------------
    # AUTO-CORRECT — runs the full two-pass NLP pipeline
    # ----------------------------------------------------

    def _auto_correct(self):
        """
        Called when user clicks "Auto-Correct All".
        Runs the complete spell_correct() pipeline from the backend:
          Pass 1: Fix non-word errors using Noisy Channel Model
          Pass 2: Fix real-word errors using bigram sentence scoring
        Then replaces ALL text with the corrected version.
        """
        if not self.backend:
            return
        content = self.text.get('1.0', 'end-1c').strip()  # Get all text, remove leading/trailing whitespace
        if not content:
            return  # Nothing to correct

        # spell_correct() is the full pipeline from the notebook
        # Returns: (corrected_text, list_of_corrections_made)
        corrected, corrections = self.backend['spell_correct'](content)

        # Replace the entire text editor content with the corrected version
        self.text.delete('1.0', tk.END)      # Clear everything
        self.text.insert('1.0', corrected)   # Insert the corrected text

        # Re-run highlighting to show any words that are STILL misspelled
        # (some words might not have good candidates, so they remain wrong)
        self._check_spelling()

        # Update status bar to show what was changed
        if corrections:
            # Build a summary like: 'teh'→'the', 'mney'→'money'
            details = ', '.join(f"'{c['original']}'→'{c['corrected']}'" for c in corrections)
            self.status_var.set(f"Auto-corrected {len(corrections)} word(s): {details}")
        else:
            self.status_var.set("No corrections needed.")


# =============================================================
# 3. MAIN — Entry point: creates the window and starts the app
# =============================================================

if __name__ == '__main__':
    # This block runs only when you execute this file directly
    # (not when it's imported as a module)

    root = tk.Tk()                  # Create the main application window
    app = SpellCheckerGUI(root)     # Build the GUI inside it
    root.mainloop()                 # Start the event loop — keeps the window open
    # mainloop() listens for events (clicks, keystrokes, etc.) and dispatches
    # them to the right handler functions. It runs forever until the window is closed.