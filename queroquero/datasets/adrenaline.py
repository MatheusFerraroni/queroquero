from __future__ import annotations

from ._conversation_zip import ConversationZipAdapter


class AdrenalineAdapter(ConversationZipAdapter):
    dataset_id = "adrenaline"


ADAPTER = AdrenalineAdapter()
