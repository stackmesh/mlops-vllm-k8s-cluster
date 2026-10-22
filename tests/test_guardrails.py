from mlops_cluster.guardrails import enforce_citations


def test_known_citation_is_allowed():
    result = enforce_citations("The quota is two GPUs [budget-quota].", {"budget-quota"})
    assert result.allowed
    assert result.citations == ("budget-quota",)


def test_unknown_citation_is_rejected():
    result = enforce_citations("Unlimited capacity [fake-doc].", {"budget-quota"})
    assert result.allowed is False
    assert result.unknown == ("fake-doc",)


def test_missing_citation_is_rejected():
    result = enforce_citations("The quota is two GPUs.", {"budget-quota"})
    assert result.allowed is False
    assert "omitted" in result.reason


def test_empty_context_is_rejected():
    result = enforce_citations("Anything [budget-quota].", set())
    assert result.allowed is False
