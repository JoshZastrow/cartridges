from cartridges.training.loops import ends_in_loop, loop_rate


def test_clean_page_does_not_loop():
    assert not ends_in_loop("Woke up early and walked the dog before work.")


def test_repeated_phrase_at_end_loops():
    text = "I went to the store and bought milk and then I went home and then I went home and then I went home"
    assert ends_in_loop(text)


def test_single_word_loop():
    assert ends_in_loop("so tired today today today today")


def test_loop_cut_off_mid_phrase_by_token_cap():
    text = "the end of the day the end of the day the end of the day the end of"
    assert ends_in_loop(text)


def test_loop_ignores_case_and_punctuation():
    assert ends_in_loop("Then I went home. Then I went home, then I went home")


def test_two_repeats_is_not_a_loop():
    assert not ends_in_loop("I said no, I said no")


def test_repetition_in_the_middle_does_not_count():
    assert not ends_in_loop("very very very tired but then I got up and made coffee")


def test_phrase_longer_than_max_is_ignored():
    phrase = " ".join(f"w{i}" for i in range(15))
    assert not ends_in_loop(" ".join([phrase] * 3), max_phrase_words=12)


def test_empty_and_none():
    assert not ends_in_loop("")
    assert not ends_in_loop(None)


def test_loop_rate():
    assert loop_rate(["fine page", "go go go go", "another fine page", "last page"]) == 0.25
    assert loop_rate(["go go go go", None]) == 1.0  # failed inferences are excluded
    assert loop_rate([]) == 0.0


def test_loop_rate_forgives_loops_that_are_in_the_reference():
    texts = ["Repair, repair, repair.", "go go go go"]
    references = ["Repair, repair, repair.", "go"]
    assert loop_rate(texts, references) == 0.5
