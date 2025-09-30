from typing import List, Dict

from manager import CompletionMessage


class LLMMessageFormater:

    def __init__(self, history_msg_limit=20):
        self.history_msg_limit = history_msg_limit

    async def format(self, session) -> List:
        """大模型消息格式化
        """
        llm_messages = []
        messages = session.messages
        if messages:
            messages = messages[-self.history_msg_limit:]
            for message in messages:
                llm_messages.append({
                    "role": message.role,
                    "content": message.content,
                })

        return llm_messages


class RespMessageFormater:

    @staticmethod
    async def format(session_id: str, message: CompletionMessage) -> Dict:
        return {
            "session_id": session_id,
            "message_id": message.id,
            "message": message.content,
            "timestamp": message.timestamp,
        }
