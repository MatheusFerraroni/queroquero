from __future__ import annotations

from ._conversation_zip import ConversationZipAdapter


class OuterSpaceAdapter(ConversationZipAdapter):
    dataset_id = "outerspace"


ADAPTER = OuterSpaceAdapter()
