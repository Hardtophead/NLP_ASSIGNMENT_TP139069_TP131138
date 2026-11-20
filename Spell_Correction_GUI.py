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

# ══════════════════════════════════════════════════════════════
# 1. NLP BACKEND  (copied from your notebook — no changes)
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# 2. GUI
# ══════════════════════════════════════════════════════════════

class SpellCheckerGUI:
    MAX_CHARS = 500

    def __init__(self, root):
        self.root = root
        self.root.title("Spelling Correction System — CT052-3-M-NLP")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.backend = None           # filled after loading
        self.misspelled_ranges = {}   # tag_name -> word
        self._selected_tag = None

        self._build_ui()
        self._start_loading()

    # ── UI construction ──────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Status.TLabel', font=('Segoe UI', 10))

        # --- Top bar ---
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Spelling Correction System", style='Title.TLabel').pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Loading models…")
        ttk.Label(top, textvariable=self.status_var, style='Status.TLabel').pack(side=tk.RIGHT)

        # --- Main paned area ---
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # LEFT: dictionary panel
        left = ttk.LabelFrame(pane, text="Dictionary (Brown Corpus)", padding=6)
        pane.add(left, weight=1)

        # search bar
        sf = ttk.Frame(left)
        sf.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(sf, text="Search:").pack(side=tk.LEFT)
        self.dict_search_var = tk.StringVar()
        self.dict_search_var.trace_add('write', self._filter_dict)
        ttk.Entry(sf, textvariable=self.dict_search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self.dict_list = tk.Listbox(left, font=('Consolas', 10))
        dict_scroll = ttk.Scrollbar(left, command=self.dict_list.yview)
        self.dict_list.configure(yscrollcommand=dict_scroll.set)
        dict_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.dict_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.dict_count_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.dict_count_var).pack(anchor=tk.W, pady=(4, 0))

        # RIGHT panel
        right = ttk.Frame(pane, padding=6)
        pane.add(right, weight=2)

        # text editor
        editor_frame = ttk.LabelFrame(right, text="Text Editor (max 500 characters)", padding=4)
        editor_frame.pack(fill=tk.BOTH, expand=True)

        self.text = tk.Text(editor_frame, font=('Segoe UI', 12), wrap=tk.WORD, height=8)
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.bind('<KeyRelease>', self._on_key)
        self.text.bind('<Button-1>', self._on_click)

        self.char_count_var = tk.StringVar(value="0 / 500 characters")
        ttk.Label(editor_frame, textvariable=self.char_count_var).pack(anchor=tk.E)

        # buttons
        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=6)
        self.check_btn = ttk.Button(btn_frame, text="Check Spelling", command=self._check_spelling, state=tk.DISABLED)
        self.check_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.correct_btn = ttk.Button(btn_frame, text="Auto-Correct All", command=self._auto_correct, state=tk.DISABLED)
        self.correct_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Clear", command=self._clear).pack(side=tk.LEFT)

        # suggestions panel
        sug_frame = ttk.LabelFrame(right, text="Suggestions (click a highlighted word)", padding=4)
        sug_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 0))

        cols = ('candidate', 'edit_dist', 'channel', 'prior', 'score')
        self.sug_tree = ttk.Treeview(sug_frame, columns=cols, show='headings', height=6)
        self.sug_tree.heading('candidate',  text='Candidate')
        self.sug_tree.heading('edit_dist',  text='Edit Dist')
        self.sug_tree.heading('channel',    text='P(x|w)')
        self.sug_tree.heading('prior',      text='P(w)')
        self.sug_tree.heading('score',      text='Score')
        self.sug_tree.column('candidate', width=120)
        self.sug_tree.column('edit_dist', width=70, anchor=tk.CENTER)
        self.sug_tree.column('channel',   width=100, anchor=tk.E)
        self.sug_tree.column('prior',     width=100, anchor=tk.E)
        self.sug_tree.column('score',     width=100, anchor=tk.E)
        self.sug_tree.pack(fill=tk.BOTH, expand=True)
        self.sug_tree.bind('<Double-1>', self._apply_suggestion)

        self.sug_label_var = tk.StringVar(value="")
        ttk.Label(sug_frame, textvariable=self.sug_label_var).pack(anchor=tk.W, pady=(2, 0))

        # configure highlight tag
        self.text.tag_configure('misspelled', background='#FFCCCC', underline=True)

    # ── Loading ───────────────────────────────────────────────

    def _start_loading(self):
        def load():
            self.backend = build_backend(status_callback=lambda m: self.root.after(0, self.status_var.set, m))
            self.root.after(0, self._on_loaded)
        threading.Thread(target=load, daemon=True).start()

    def _on_loaded(self):
        self.status_var.set(f"Ready — {len(self.backend['VOCABULARY']):,} words in dictionary")
        self.check_btn.config(state=tk.NORMAL)
        self.correct_btn.config(state=tk.NORMAL)

        self.sorted_vocab = sorted(self.backend['VOCABULARY'])
        self._populate_dict(self.sorted_vocab)

    def _populate_dict(self, words):
        self.dict_list.delete(0, tk.END)
        for w in words:
            self.dict_list.insert(tk.END, w)
        self.dict_count_var.set(f"{len(words):,} words")

    def _filter_dict(self, *_):
        query = self.dict_search_var.get().lower().strip()
        if not self.backend:
            return
        if not query:
            self._populate_dict(self.sorted_vocab)
        else:
            filtered = [w for w in self.sorted_vocab if w.startswith(query)]
            self._populate_dict(filtered)

    # ── Editor helpers ────────────────────────────────────────

    def _on_key(self, event=None):
        content = self.text.get('1.0', 'end-1c')
        length = len(content)
        if length > self.MAX_CHARS:
            self.text.delete(f'1.0+{self.MAX_CHARS}c', tk.END)
            length = self.MAX_CHARS
        self.char_count_var.set(f"{length} / {self.MAX_CHARS} characters")

    def _clear(self):
        self.text.delete('1.0', tk.END)
        self._clear_highlights()
        self.sug_tree.delete(*self.sug_tree.get_children())
        self.sug_label_var.set("")
        self.char_count_var.set("0 / 500 characters")

    # ── Spell checking ────────────────────────────────────────

    def _clear_highlights(self):
        for tag in list(self.misspelled_ranges.keys()):
            self.text.tag_delete(tag)
        self.misspelled_ranges.clear()
        self.text.tag_remove('misspelled', '1.0', tk.END)
        self._selected_tag = None

    def _check_spelling(self):
        if not self.backend:
            return
        self._clear_highlights()
        self.sug_tree.delete(*self.sug_tree.get_children())
        self.sug_label_var.set("")

        content = self.text.get('1.0', 'end-1c')
        VOCAB = self.backend['VOCABULARY']

        # find each word by regex and check
        error_count = 0
        for m in re.finditer(r'[a-zA-Z]+', content):
            word = m.group().lower()
            if word not in VOCAB:
                start_idx = f'1.0+{m.start()}c'
                end_idx   = f'1.0+{m.end()}c'
                tag = f'err_{error_count}'
                self.text.tag_add('misspelled', start_idx, end_idx)
                self.text.tag_add(tag, start_idx, end_idx)
                self.text.tag_bind(tag, '<Button-1>', lambda e, t=tag: self._on_word_click(t))
                self.misspelled_ranges[tag] = word
                error_count += 1

        if error_count == 0:
            self.status_var.set("No spelling errors found!")
        else:
            self.status_var.set(f"Found {error_count} misspelled word(s) — click one for suggestions")

    def _on_click(self, event):
        # check if clicked position is inside any error tag
        idx = self.text.index(f'@{event.x},{event.y}')
        for tag in self.misspelled_ranges:
            ranges = self.text.tag_ranges(tag)
            if ranges:
                start, end = ranges[0], ranges[1]
                if self.text.compare(idx, '>=', start) and self.text.compare(idx, '<=', end):
                    self._on_word_click(tag)
                    return

    def _on_word_click(self, tag):
        if not self.backend:
            return
        word = self.misspelled_ranges.get(tag)
        if not word:
            return
        self._selected_tag = tag
        self.sug_tree.delete(*self.sug_tree.get_children())

        _, scored = self.backend['correct_nonword'](word)
        med = self.backend['min_edit_distance']

        self.sug_label_var.set(f"Suggestions for \"{word}\" — double-click to apply")
        for c, pxw, pw, score in scored:
            dist = med(word, c)
            self.sug_tree.insert('', tk.END, values=(
                c, dist, f'{pxw:.6f}', f'{pw:.2e}', f'{score:.2e}'
            ))

    def _apply_suggestion(self, event):
        sel = self.sug_tree.selection()
        if not sel or not self._selected_tag:
            return
        candidate = self.sug_tree.item(sel[0])['values'][0]
        tag = self._selected_tag
        ranges = self.text.tag_ranges(tag)
        if ranges:
            start, end = ranges[0], ranges[1]
            self.text.delete(start, end)
            self.text.insert(start, candidate)
            self.text.tag_delete(tag)
            del self.misspelled_ranges[tag]
        self.sug_tree.delete(*self.sug_tree.get_children())
        self.sug_label_var.set(f"Applied: {candidate}")
        self._selected_tag = None
        # re-check to update highlights
        self._check_spelling()

    # ── Auto-correct ──────────────────────────────────────────

    def _auto_correct(self):
        if not self.backend:
            return
        content = self.text.get('1.0', 'end-1c').strip()
        if not content:
            return

        corrected, corrections = self.backend['spell_correct'](content)
        self.text.delete('1.0', tk.END)
        self.text.insert('1.0', corrected)

        # re-highlight anything still misspelled
        self._check_spelling()

        if corrections:
            details = ', '.join(f"'{c['original']}'→'{c['corrected']}'" for c in corrections)
            self.status_var.set(f"Auto-corrected {len(corrections)} word(s): {details}")
        else:
            self.status_var.set("No corrections needed.")


# ══════════════════════════════════════════════════════════════
# 3. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    root = tk.Tk()
    app = SpellCheckerGUI(root)
    root.mainloop()