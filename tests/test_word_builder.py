from modules.word_builder import WordBuilder


def builder():
    return WordBuilder(
        hold_time_seconds=1.0,
        delete_labels={"delete"},
        space_labels={"space"},
        clear_labels={"clear"},
        ignore_labels={"nothing"},
    )


def test_hold_to_accept_adds_once():
    word_builder = builder()
    assert word_builder.observe("A", 0.9, now=0.0) is None
    assert word_builder.observe("A", 0.9, now=0.5) is None
    assert word_builder.observe("A", 0.9, now=1.1) == "Added A"
    assert word_builder.current_word == "A"
    assert word_builder.observe("A", 0.9, now=2.4) is None
    assert word_builder.current_word == "A"


def test_commit_and_delete():
    word_builder = builder()
    word_builder.accept_label("H")
    word_builder.accept_label("I")
    assert word_builder.commit_word() == "Committed HI"
    assert word_builder.sentence == "HI "
    word_builder.delete_last()
    assert word_builder.sentence == ""
