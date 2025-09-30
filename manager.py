import time
import asyncio
import datetime
import logging
import traceback
from enum import Enum
from typing import List, Dict

import shortuuid
from fastapi import WebSocket


logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    CREATED = 'created'
    RUNNING = 'running'
    ERROR = 'error'
    CANCELLED = 'cancelled'
    COMPLETED = 'completed'


class Role(str, Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'


class BaseMessage:
    pass


class CompletionMessage(BaseMessage):

    def __init__(self,
                 role: Role,
                 content: str | None,
                 name: str | None = None):
        self.id = shortuuid.uuid()
        self.name = name
        self.role = role
        self.content = content
        self.timestamp = time.time()


class Session:
    """session，每个session中有多条task（message）
    """

    def __init__(self,
                 session_id: str | None = None,
                 task: asyncio.Task | None = None,
                 status: SessionStatus = SessionStatus.CREATED):
        self.id = session_id or shortuuid.uuid()
        self.task: asyncio.Task = task
        self.cancel_event = asyncio.Event()
        self.messages: List[CompletionMessage] = []
        self.status: SessionStatus = status
        self.is_cancelled = False
        self.start_time = datetime.datetime.now().isoformat()
        self.end_time = None


class SessionManager:

    def __init__(self):
        self.sessions: Dict[str, Session] = {}

    async def create_session(self, session_id):
        """创建会话
        """
        if not session_id:
            session = Session()
            self.sessions[session.id] = session
        elif not self.sessions.get(session_id):
            session = Session(session_id)
            self.sessions[session.id] = session
        else:
            session = self.sessions.get(session_id)

        logger.info(f"会话 {session_id} 创建成功")
        return session

    async def get_session(self, session_id: str) -> Session:
        """获取会话信息
        """
        return self.sessions.get(session_id)

    async def cancel_session(self, session_id=None):
        """取消会话
        """
        session_list = []
        if session_id:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"会话 {session_id} 不存在，取消失败")

            session_list.append(session)
        else:
            if self.sessions:
                session_list = self.sessions.values()

        if session_list:
            enable_cancel_status = [SessionStatus.CREATED, SessionStatus.RUNNING]
            for session in session_list:
                if session.status in enable_cancel_status:
                    session.cancel_event.set()
                    session.is_cancelled = True
                    session.status = SessionStatus.CANCELLED
                    session.end_time = datetime.datetime.now().isoformat()

            return True
        return False


class WebSocketManager:
    def __init__(self, client_id=None):
        self.client_id = client_id or shortuuid.uuid()
        self.connections: List[WebSocket] = []
        self.session_manager: Dict[str, SessionManager] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        try:
            await websocket.accept()
        except Exception as e:
            logger.error(traceback.format_exc())
            raise

        self.connections.append(websocket)

        if not self.client_id or self.client_id not in self.session_manager:
            self.client_id = client_id
            self.session_manager[client_id] = SessionManager()

        logger.info(f"客户端 {self.client_id} 已连接")

    async def disconnect(self, websocket: WebSocket):
        # 移除WebSocket连接
        self.connections.remove(websocket)

        # 清空sessions
        session_manager = self.session_manager.pop(self.client_id)
        if session_manager:
            session_manager.sessions = []

        logger.info(f"客户端 {self.client_id} 连接已断开")

    @staticmethod
    async def send_text_message(message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    async def send_json_message(message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(traceback.format_exc())
            raise

    def check_session(self, session_id):
        if self.client_id not in self.session_manager:
            logger.warning(f"客户端 {self.client_id} 连接不存在")
            raise ValueError(f"客户端 {self.client_id} 连接不存在")

        session_manager = self.session_manager[self.client_id]
        if session_id not in session_manager.sessions:
            logger.warning(f"客户端 {self.client_id} 会话 {session_id} 不存在")
            raise ValueError(f"客户端 {self.client_id} 会话 {session_id} 不存在")

        return session_manager.sessions[session_id]

    def add_history(self, session_id: str, message: CompletionMessage):
        """添加历史消息
        """
        try:
            session = self.check_session(session_id)
            session.messages.append(message)
            return
        except Exception as e:
            logger.error(traceback.format_exc())
            raise

    def get_history(self, session_id: str) -> List[CompletionMessage]:
        """获取历史消息
        """
        try:
            session = self.check_session(session_id)
            return session.messages
        except Exception as e:
            logger.error(traceback.format_exc())
            raise
