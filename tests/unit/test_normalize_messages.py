from __future__ import annotations

from medflow.providers.base import Message, normalize_messages


def test_normalize_dict_to_message() -> None:
    msgs = normalize_messages([{"role": "user", "content": "hi"}])
    assert len(msgs) == 1
    assert msgs[0].role == "user"
    assert msgs[0].content == "hi"


def test_normalize_mixed() -> None:
    msgs = normalize_messages([Message(role="system", content="s"), {"role": "user", "content": "u"}])
    assert msgs[0].role == "system"
    assert msgs[1].content == "u"
