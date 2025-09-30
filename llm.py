import logging
import traceback

from dashscope.aigc.generation import AioGeneration


logger = logging.getLogger(__name__)


class DashScopeLLMClient:

    def __init__(self, base_url, api_key):
        self.base_url: str = base_url
        self.api_key: str = api_key
        self.default_model = 'qwen3-max'

    async def chat_stream(self, model, stream, messages):
        try:
            # 使用异步的方式调用大模型，不会阻塞接口
            response = await AioGeneration.call(
                api_key=self.api_key,
                model=model,
                stream=stream,
                messages=messages,
                result_format="message",
                incremental_output=stream
            )
            return response
        except Exception as e:
            logger.error(traceback.format_exc())