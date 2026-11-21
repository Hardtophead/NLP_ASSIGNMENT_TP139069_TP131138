"""
Spelling Correction System — Streamlit Web App
CT052-3-M-NLP Assignment

Requirements fulfilled:
  1. Sorted dictionary list with search facility
  2. 500-character text editor
  3. Highlight misspelled words
  4. Click on misspelled word → suggest corrections with edit distances
  5. Non-word AND real-word error correction (two-pass pipeline)

Run with:  streamlit run spell_checker_app.py
Requires:  spell-errors.txt in the same directory
"""

import streamlit as st
import re
import math
import os
from collections import Counter, defaultdict

import nltk
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import brown
from nltk.probability import FreqDist
from nltk.lm import Laplace as LaplaceLM
from nltk.lm.preprocessing import padded_everygram_pipeline


# ══════════════════════════════════════════════════════════════
# 1. NLP BACKEND  (cached so it only loads once)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading NLP models — this takes ~15 seconds on first run...")
def build_backend():
    """Build all NLP models. Returns a dict of everything the app needs."""

    # --- Corpus ---
    raw_words = brown.words()
    corpus_words = [w.lower() for w in raw_words if w.isalpha()]
    word_counts = FreqDist(corpus_words)
    VOCABULARY = set(word_counts.keys())
    TOTAL_WORDS = word_counts.N()
    VOCAB_SIZE = word_counts.B()

    # --- Word probability (Laplace smoothing) ---
    def word_probability(word):
        return (word_counts[word] + 1) / (TOTAL_WORDS + VOCAB_SIZE)

    # --- Bigram Language Model ---
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

    # --- Damerau-Levenshtein Edit Distance ---
    def min_edit_distance(source, target, sub_cost=1):
        n, m = len(source), len(target)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if source[i - 1] == target[j - 1] else sub_cost
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
                if i > 1 and j > 1 and source[i - 1] == target[j - 2] and source[i - 2] == target[j - 1]:
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
        return dp[n][m]

    # --- Candidate generation ---
    def edits_one(word):
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [L + R[1:]           for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces   = [L + c + R[1:]       for L, R in splits if R for c in letters]
        inserts    = [L + c + R            for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits_two(word):
        return {e2 for e1 in edits_one(word) for e2 in edits_one(e1) if e2 in VOCABULARY}

    # --- Confusion matrices from spell-errors.txt ---
    def identify_edit(source, target):
        ls, lt = len(source), len(target)
        if ls == lt - 1:
            for i in range(lt):
                if source == target[:i] + target[i + 1:]:
                    prev = target[i - 1] if i > 0 else '#'
                    return ('del', prev, target[i])
        if ls == lt + 1:
            for i in range(ls):
                if target == source[:i] + source[i + 1:]:
                    prev = source[i - 1] if i > 0 else '#'
                    return ('ins', prev, source[i])
        if ls == lt:
            diffs = [(i, source[i], target[i]) for i in range(ls) if source[i] != target[i]]
            if len(diffs) == 1:
                return ('sub', diffs[0][1], diffs[0][2])
            if len(diffs) == 2:
                i, j = diffs[0][0], diffs[1][0]
                if j == i + 1 and source[i] == target[j] and source[j] == target[i]:
                    return ('trans', source[i], source[j])
        return None

    char_bigram_counts = Counter()
    char_unigram_counts = Counter()
    for w in corpus_words:
        for ch in w:
            char_unigram_counts[ch] += 1
        for i in range(len(w) - 1):
            char_bigram_counts[w[i] + w[i + 1]] += 1

    del_matrix   = defaultdict(lambda: defaultdict(lambda: 1))
    ins_matrix   = defaultdict(lambda: defaultdict(lambda: 1))
    sub_matrix   = defaultdict(lambda: defaultdict(lambda: 1))
    trans_matrix = defaultdict(lambda: defaultdict(lambda: 1))

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
            return del_matrix[c1][c2] / (char_bigram_counts.get(c1 + c2, 1) + 26)
        elif op == 'ins':
            return ins_matrix[c1][c2] / (char_unigram_counts.get(c1, 1) + 26)
        elif op == 'sub':
            return sub_matrix[c1][c2] / (char_unigram_counts.get(c2, 1) + 26)
        elif op == 'trans':
            return trans_matrix[c1][c2] / (char_bigram_counts.get(c1 + c2, 1) + 26)
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

    return {
        'VOCABULARY': VOCABULARY,
        'sorted_vocab': sorted(VOCABULARY),
        'word_counts': word_counts,
        'min_edit_distance': min_edit_distance,
        'correct_nonword': correct_nonword,
        'correct_realword': correct_realword,
        'spell_correct': spell_correct,
        'get_candidates': get_candidates,
        'channel_probability': channel_probability,
        'word_probability': word_probability,
        'edits_one': edits_one,
        'TOTAL_WORDS': TOTAL_WORDS,
        'VOCAB_SIZE': VOCAB_SIZE,
    }


# ══════════════════════════════════════════════════════════════
# 2. STREAMLIT APP
# ══════════════════════════════════════════════════════════════

def find_misspelled(text, vocab):
    """Find all misspelled words and their positions in the text."""
    errors = []
    for m in re.finditer(r'[a-zA-Z]+', text):
        word = m.group().lower()
        if word not in vocab:
            errors.append({
                'word': m.group(),           # original case
                'word_lower': word,          # lowercase for lookup
                'start': m.start(),
                'end': m.end()
            })
    return errors


def highlight_text(text, errors):
    """
    Build an HTML string with misspelled words highlighted in red.
    Uses <mark> tags for highlighting.
    """
    if not errors:
        return text

    html_parts = []
    last_end = 0
    for err in sorted(errors, key=lambda e: e['start']):
        # Add the text before this error
        html_parts.append(text[last_end:err['start']])
        # Add the highlighted error word
        html_parts.append(
            f'<mark style="background-color: #FFCCCC; padding: 2px 4px; '
            f'border-radius: 3px; border-bottom: 2px solid #E53935;">'
            f'{text[err["start"]:err["end"]]}</mark>'
        )
        last_end = err['end']
    html_parts.append(text[last_end:])  # remaining text after last error

    return ''.join(html_parts)


# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Spelling Correction System — NLP",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Spelling Correction System")
st.caption("CT052-3-M-NLP — Noisy Channel Model + Bigram Language Model | Brown Corpus")

# ── Load backend ─────────────────────────────────────────────

backend = build_backend()
VOCAB = backend['VOCABULARY']
sorted_vocab = backend['sorted_vocab']

# ── Initialize session state ─────────────────────────────────

if 'selected_word' not in st.session_state:
    st.session_state.selected_word = None
if 'corrections_made' not in st.session_state:
    st.session_state.corrections_made = []
if 'check_results' not in st.session_state:
    st.session_state.check_results = None

# ── Layout: two columns ─────────────────────────────────────

left_col, right_col = st.columns([1, 3])

# ── LEFT COLUMN: Dictionary ─────────────────────────────────

with left_col:
    st.subheader("📖 Dictionary")
    st.caption(f"Brown Corpus — {len(VOCAB):,} words")

    # Search bar
    search_query = st.text_input("Search dictionary", placeholder="Type to filter...")

    # Filter vocabulary
    if search_query:
        filtered = [w for w in sorted_vocab if w.startswith(search_query.lower().strip())]
    else:
        filtered = sorted_vocab

    st.caption(f"Showing {len(filtered):,} words")

    # Display as a scrollable list (show first 500 to keep it fast)
    display_words = filtered[:500]
    if len(filtered) > 500:
        st.info(f"Showing first 500 of {len(filtered):,} matches. Narrow your search to see more.")

    # Use a text area styled as a list for scrollability
    dict_text = "\n".join(display_words)
    st.text_area("Word list", value=dict_text, height=400, disabled=True, label_visibility="collapsed")

# ── RIGHT COLUMN: Editor + Spelling ─────────────────────────

with right_col:
    st.subheader("✍️ Text Editor")

    # Text input area (500 char max)
    user_text = st.text_area(
        "Enter your text (max 500 characters)",
        max_chars=500,
        height=150,
        placeholder="Type or paste your text here to check for spelling errors..."
    )

    char_count = len(user_text) if user_text else 0
    st.caption(f"{char_count} / 500 characters")

    # ── Buttons row ──────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    with btn_col1:
        check_clicked = st.button("🔍 Check Spelling", type="primary", use_container_width=True)
    with btn_col2:
        correct_clicked = st.button("✨ Auto-Correct All", use_container_width=True)

    # ── Check Spelling ───────────────────────────────────────

    if check_clicked and user_text:
        errors = find_misspelled(user_text, VOCAB)
        st.session_state.check_results = errors
        st.session_state.selected_word = None
        st.session_state.corrections_made = []

    if correct_clicked and user_text:
        corrected_text, corrections = backend['spell_correct'](user_text)
        st.session_state.corrections_made = corrections
        st.session_state.check_results = 'auto_corrected'
        st.session_state.corrected_text = corrected_text

    # ── Display results ──────────────────────────────────────

    if st.session_state.check_results is not None:

        # --- Auto-Correct Results ---
        if st.session_state.check_results == 'auto_corrected':
            corrected_text = st.session_state.corrected_text
            corrections = st.session_state.corrections_made

            if corrections:
                st.success(f"✅ Auto-corrected {len(corrections)} word(s)")

                # Show what was corrected
                st.markdown("**Corrections made:**")
                for c in corrections:
                    badge_color = "#E3F2FD" if c['type'] == 'non-word' else "#FFF3E0"
                    type_label = "Non-word" if c['type'] == 'non-word' else "Real-word"
                    st.markdown(
                        f'<span style="background:{badge_color}; padding:2px 8px; '
                        f'border-radius:12px; font-size:0.85em;">{type_label}</span> '
                        f'**{c["original"]}** → **{c["corrected"]}** '
                        f'(edit distance: {c["edit_distance"]})',
                        unsafe_allow_html=True
                    )

                # Show corrected text
                st.markdown("---")
                st.markdown("**Corrected text:**")
                st.markdown(
                    f'<div style="background:#F1F8E9; padding:16px; border-radius:8px; '
                    f'font-size:1.1em; line-height:1.6;">{corrected_text}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.success("✅ No corrections needed — text looks good!")

        # --- Check Spelling Results ---
        else:
            errors = st.session_state.check_results

            if not errors:
                st.success("✅ No spelling errors found!")
            else:
                st.warning(f"Found **{len(errors)}** misspelled word(s)")

                # Show highlighted text
                highlighted = highlight_text(user_text, errors)
                st.markdown(
                    f'<div style="background:#FAFAFA; padding:16px; border-radius:8px; '
                    f'font-size:1.1em; line-height:1.8;">{highlighted}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("---")

                # Let user select which word to get suggestions for
                misspelled_words = list(dict.fromkeys([e['word_lower'] for e in errors]))
                st.markdown("**Click a misspelled word to see suggestions:**")
                # Create a row of buttons for each misspelled word
                word_cols = st.columns(min(len(misspelled_words), 6))
                for idx, word in enumerate(misspelled_words):
                    col_idx = idx % min(len(misspelled_words), 6)
                    with word_cols[col_idx]:
                        if st.button(f"❌ {word}", key=f"word_{idx}", use_container_width=True):
                            st.session_state.selected_word = word

                # ── Suggestions table ────────────────────────
                if st.session_state.selected_word:
                    sel_word = st.session_state.selected_word
                    st.markdown(f'### Suggestions for "**{sel_word}**"')

                    _, scored = backend['correct_nonword'](sel_word)
                    med = backend['min_edit_distance']

                    if scored:
                        # Build a table
                        table_data = []
                        for c, pxw, pw, score in scored:
                            dist = med(sel_word, c)
                            table_data.append({
                                'Candidate': c,
                                'Edit Distance': dist,
                                'P(x|w)': f'{pxw:.6f}',
                                'P(w)': f'{pw:.2e}',
                                'Score': f'{score:.2e}'
                            })

                        st.table(table_data)
                        st.caption(
                            f"Best suggestion: **{scored[0][0]}** "
                            f"(edit distance: {med(sel_word, scored[0][0])}, "
                            f"score: {scored[0][3]:.2e})"
                        )
                    else:
                        st.info("No suggestions found for this word.")

# ── Footer ───────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#888; font-size:0.85em;">'
    'Spelling Correction System — CT052-3-M-NLP Assignment<br>'
    'Noisy Channel Model: ŵ = argmax P(x|w) × P(w) | '
    'Damerau-Levenshtein Edit Distance | Bigram Language Model (NLTK Laplace)<br>'
    'Corpus: Brown Corpus (981K words, 40K vocabulary) | '
    'Error data: Norvig spell-errors.txt (39K pairs)'
    '</div>',
    unsafe_allow_html=True
)
