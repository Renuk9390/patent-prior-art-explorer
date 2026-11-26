from patentsearch.ingest.patents import build_passages, first_independent_claim, split_claims

CLAIMS = """What is claimed is:
1. A method for training a neural network, comprising:
 receiving training data; and
 updating weights in step 3. using gradient descent.
2. The method of claim 1, wherein the training data comprises images.
3. The method according to claim 1, further comprising pruning the network.
4. (canceled)
5. A system comprising a processor configured to perform the method.
6. The system as recited in claim 5, wherein the processor is a GPU.
"""


def test_split_claims_numbers_and_dependency():
    claims = split_claims(CLAIMS)
    assert [c.number for c in claims] == [1, 2, 3, 5, 6]          # 4 cancelled
    assert [c.independent for c in claims] == [True, False, False, True, False]
    assert "step 3. using gradient descent" in claims[0].text   # not split at "3."


def test_build_passages_labels():
    passages = build_passages("US-1-B2", "An abstract about neural networks.", CLAIMS)
    labels = [p["label"] for p in passages]
    assert labels == ["Abstract", "Claim 1", "Claims 2-3", "Claim 5", "Claim 6"]
    assert passages[0]["passage_id"] == "US-1-B2#0"
    assert passages[2]["text"].startswith("2. The method of claim 1")


def test_long_claim_is_split_into_parts():
    long_claim = "1. A method comprising " + " ".join(["step"] * 600) + "."
    passages = build_passages("US-2-B1", None, long_claim, max_words=250)
    assert [p["label"] for p in passages] == ["Claim 1 (part 1)", "Claim 1 (part 2)", "Claim 1 (part 3)"]
    assert all(len(p["text"].split()) <= 250 for p in passages)


def test_dependent_claims_grouped_by_size():
    claims = "1. A device. " + " ".join(
        f"{n}. The device of claim 1, wherein " + "x " * 60 + "end." for n in range(2, 8)
    )
    labels = [p["label"] for p in build_passages("US-3-B2", None, claims, group_words=200)]
    assert labels == ["Claim 1", "Claims 2-4", "Claims 5-7"]   # 3 x 67 words >= 200


def test_claims_on_one_line_are_split():
    claims = split_claims("1. A device comprising a sensor. 2. The device of claim 1, wherein the sensor is optical.")
    assert [(c.number, c.independent) for c in claims] == [(1, True), (2, False)]


def test_first_independent_claim_and_empty_input():
    assert first_independent_claim(CLAIMS).startswith("A method for training")
    assert split_claims("") == [] and build_passages("US-4", None, None) == []


def test_long_dependent_claim_is_split():
    claims = "1. A device. 2. The device of claim 1, wherein " + " ".join(["x"] * 600) + "."
    passages = build_passages("US-5-B2", None, claims, max_words=250)
    assert [p["label"] for p in passages] == [
        "Claim 1", "Claim 2 (part 1)", "Claim 2 (part 2)", "Claim 2 (part 3)"]
    assert max(len(p["text"].split()) for p in passages) <= 250
