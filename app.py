import json
import logging
import asyncio
import traceback
from http import HTTPStatus

from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from llm import DashScopeLLMClient
from manager import WebSocketManager
from manager import Role, CompletionMessage, SessionStatus
from formater import LLMMessageFormater, RespMessageFormater


logger = logging.getLogger(__name__)


BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
API_KEY = 'xxx'


app = FastAPI()


websocket_manager = WebSocketManager()
llm_client = DashScopeLLMClient(base_url=BASE_URL, api_key=API_KEY)


async def completion(websocket, session, client_args):
    llm_msg_formater = LLMMessageFormater()
    resp_msg_formater = RespMessageFormater()

    session_id = session.id

    user_message = client_args.get("message", "")
    stream = client_args.get("stream", True)

    # 保存为历史消息
    message = CompletionMessage(
        name="user",
        role=Role.USER,
        content=user_message
    )
    websocket_manager.add_history(session_id, message)

    # 格式化llm消息
    llm_messages = await llm_msg_formater.format(session)
    completion_args = {
        "model": client_args.get("model", llm_client.default_model),
        "messages": llm_messages,
        "stream": stream
    }

    try:
        full_content = ""
        if stream:
            async for resp in await llm_client.chat_stream(**completion_args):
                if session.cancel_event.is_set():
                    return

                if resp.status_code == HTTPStatus.OK and resp.output.choices:
                    choices = resp.output.choices
                    if not choices:
                        continue

                    curr_content = choices[0].message.content
                    full_content += curr_content

                    resp_message = await resp_msg_formater.format(
                        session_id,
                        CompletionMessage(
                            role=Role.ASSISTANT,
                            content=curr_content
                        )
                    )

                    await websocket_manager.send_json_message({
                        **resp_message,
                        "status": SessionStatus.RUNNING
                    }, websocket)

                    if session.cancel_event.is_set():
                        return
                else:
                    resp_message = await resp_msg_formater.format(
                        session_id,
                        CompletionMessage(
                            role=Role.ASSISTANT,
                            content=resp.message
                        )
                    )
                    await websocket_manager.send_json_message({
                        **resp_message,
                        "status": SessionStatus.ERROR
                    }, websocket)

        else:
            resp = await llm_client.chat_stream(**completion_args)
            if resp.status_code == HTTPStatus.OK and resp.output.choices:
                full_content = resp.output.choices[0].message.content
                resp_message = await resp_msg_formater.format(
                    session_id,
                    CompletionMessage(
                        role=Role.ASSISTANT,
                        content=full_content
                    )
                )
                await websocket_manager.send_json_message({
                    **resp_message,
                    "status": SessionStatus.RUNNING
                }, websocket)

        if not session.is_cancelled:
            resp_message = await resp_msg_formater.format(
                session_id,
                CompletionMessage(
                    role=Role.ASSISTANT,
                    content=None
                )
            )
            await websocket_manager.send_json_message({
                **resp_message,
                "status": SessionStatus.COMPLETED
            }, websocket)

            message = CompletionMessage(
                name="assistant",
                role=Role.ASSISTANT,
                content=full_content
            )
            websocket_manager.add_history(session_id, message)
    except asyncio.CancelledError:
        raise
    except json.JSONDecodeError:
        resp_message = await resp_msg_formater.format(
            session_id,
            CompletionMessage(
                role=Role.SYSTEM,
                content="无效的JSON格式"
            )
        )

        await websocket_manager.send_json_message({
            **resp_message,
            "status": SessionStatus.ERROR
        }, websocket)
    except Exception as e:
        logger.error(traceback.format_exc())

        resp_message = await resp_msg_formater.format(
            session_id,
            CompletionMessage(
                role=Role.SYSTEM,
                content=f"处理消息时出错: {str(e)}"
            )
        )
        await websocket_manager.send_json_message({
            **resp_message,
            "status": SessionStatus.ERROR
        }, websocket)

        raise

@app.websocket("/ws/chat/{client_id}")
async def websocket_chat(websocket: WebSocket, client_id: str):
    try:
        # WebSoket连接，创建session_manager，一个client_id对应一个session_manager
        await websocket_manager.connect(websocket, client_id)
    except Exception as e:
        logger.error(traceback.format_exc())
        return

    session_manager = websocket_manager.session_manager.get(client_id)
    if not session_manager:
        return

    resp_msg_formater = RespMessageFormater()

    try:
        while True:
            # 客户端消息
            data = await websocket.receive_text()
            args = json.loads(data)
            # 通过session manager创建session
            session_id = args.get('session_id')
            session = await session_manager.create_session(session_id)
            session_id = session.id
            args['session_id'] = session_id

            user_message = args.get("message", "")

            if user_message.strip() == 'cancel':
                if not session.is_cancelled:
                    await session_manager.cancel_session(session_id)

                    resp_message = await resp_msg_formater.format(
                        session_id,
                        CompletionMessage(
                            role=Role.ASSISTANT,
                            content="会话已取消"
                        )
                    )
                    await websocket_manager.send_json_message({
                        **resp_message,
                        "status": SessionStatus.CANCELLED
                    }, websocket)
            else:
                # 如果session状态是cancelled，更新为created
                if session.is_cancelled:
                    session.status = SessionStatus.CREATED
                    session.cancel_event.clear()
                    session.is_cancelled = False

            # 创建task，并在后台运行
            task = asyncio.create_task(completion(websocket, session, args))
            session.task = task

    except asyncio.CancelledError:
        await websocket_manager.disconnect(websocket)
    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 连接已断开")
        await websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(traceback.format_exc())
        await websocket_manager.disconnect(websocket)


@app.get("/")
async def get():
    """聊天页面
    """
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>聊天吧</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
            #messages { height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 20px; }
            .message { margin: 10px 0; padding: 5px; }
            .user { background-color: #e3f2fd; text-align: right; }
            .assistant { background-color: #f5f5f5; }
            .status { color: #666; font-style: italic; }
            #messageForm { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
            #messageInput { flex-grow: 1; padding: 10px; min-width: 200px; }
            #modelSelect { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }
            .form-group { display: flex; align-items: center; gap: 5px; }
            button { padding: 10px 20px; }
            .controls { display: flex; gap: 15px; align-items: center; flex-wrap: wrap; }
            .session-info { 
                background-color: #f0f8ff; 
                padding: 8px 12px; 
                border-radius: 4px; 
                margin-bottom: 10px;
                font-size: 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .session-id { 
                font-family: monospace; 
                color: #2c5282;
                font-weight: bold;
            }
            #clearSessionBtn {
                padding: 4px 8px;
                font-size: 12px;
                background-color: #e53e3e;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            }
            #clearSessionBtn:hover {
                background-color: #c53030;
            }
        </style>
        <!-- 引入 marked.js 用于解析 Markdown -->
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    </head>
    <body>
        <h1>聊天吧（简陋版）</h1>
        
        <div id="sessionInfo" class="session-info" style="display: none;">
            <span>当前会话ID: <span id="sessionIdDisplay" class="session-id"></span></span>
            <button id="clearSessionBtn" type="button">清除会话</button>
        </div>
        
        <div id="messages"></div>
        <form id="messageForm">
            <input type="text" id="messageInput" placeholder="输入消息..." required>
            <div class="controls">
                <div class="form-group">
                    <label for="modelSelected">模型:</label>
                    <select id="modelSelected">
                        <option value="qwen3-max">qwen3-max</option>
                        <option value="qwen-max">qwen-max</option>
                        <option value="qwen-plus">qwen-plus</option>
                        <option value="qwen-turbo">qwen-turbo</option>
                        <option value="qwen-flash">qwen-flash</option>
                        <option value="deepseek-v3.1">deepseek-v3.1</option>
                        <option value="qwen3-coder-plus">coder-plus</option>
                    </select>
                </div>
                <div class="form-group">
                    <label><input type="checkbox" id="streamChecked" checked>流式输出</label>
                </div>
                <button type="submit">发送</button>
                <button type="button" onclick="clearChat()">清空</button>
            </div>
        </form>
    
        <script>
            // 获取client_id
            let clientId = localStorage.getItem('clientId');
            if (!clientId) {
                clientId = 'client_' + Date.now();
                localStorage.setItem('clientId', clientId);
            }
            
            // 获取session_id
            let session_id = sessionStorage.getItem('session_id') || null;
            if (session_id) {
                document.getElementById('sessionIdDisplay').textContent = session_id;
                document.getElementById('sessionInfo').style.display = 'flex';
            }
            
            // 建立WebSocket连接
            const ws = new WebSocket(`ws://127.0.0.1:8011/ws/chat/${clientId}`);
            const messages = document.getElementById('messages');
            const messageInput = document.getElementById('messageInput');
            const streamChecked = document.getElementById('streamChecked');
            const modelSelected = document.getElementById('modelSelected');
            let currentStreamDiv = null;
            let currentStreamContent = ''; // 用于存储流式输出的完整内容
    
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
    
                if (data.status === 'running') {
                    if (streamChecked.checked === true) {
                        if (!currentStreamDiv) {
                            currentStreamDiv = addMessage('', 'assistant');
                            currentStreamContent = ''; // 初始化流式内容
                        }
                        // 累积流式内容
                        currentStreamContent += data.message;
                        // 实时解析 Markdown 内容
                        currentStreamDiv.innerHTML = marked.parse(currentStreamContent);
                    } else {
                        addMessage(data.message, 'assistant');
                    }
                } else if (data.status === 'completed' || data.status === 'cancelled') {
                    // 流式输出完成时，重新解析完整内容确保正确显示
                    if (currentStreamDiv && currentStreamContent) {
                        currentStreamDiv.innerHTML = marked.parse(currentStreamContent);
                    }
                    currentStreamDiv = null;
                    currentStreamContent = ''; // 清空累积内容
                } else if (data.status === 'error') {
                    addMessage('错误: ' + data.message, 'system');
                }
    
                messages.scrollTop = messages.scrollHeight;
                
                // 保存session_id到sessionStorage
                if (data.session_id) {
                    session_id = data.session_id;
                    sessionStorage.setItem('session_id', session_id);
                    
                    // 更新显示的session_id
                    document.getElementById('sessionIdDisplay').textContent = session_id;
                    document.getElementById('sessionInfo').style.display = 'flex';
                }
            };
    
            ws.onclose = function() {
                addMessage('连接已断开', 'system');
            };
    
            ws.onerror = function(error) {
                addMessage('连接错误: ' + error, 'system');
            };
    
            document.getElementById('messageForm').onsubmit = function(e) {
                e.preventDefault();
                const message = messageInput.value;
                if (message) {
                    addMessage(message, 'user');
                    
                    // 请求数据
                    const sendData = {
                        message: message,
                        stream: streamChecked.checked,
                        model: modelSelected.value
                    };
                    
                    // 增加session_id
                    if (session_id) {
                        sendData.session_id = session_id;
                    }
                    
                    ws.send(JSON.stringify(sendData));
                    messageInput.value = '';
                }
            };
    
            // 清除会话ID
            document.getElementById('clearSessionBtn').onclick = function() {
                session_id = null;
                sessionStorage.removeItem('session_id');
                document.getElementById('sessionInfo').style.display = 'none';
                clearChat()
            };
    
            function addMessage(message, type) {
                const div = document.createElement('div');
                div.className = 'message ' + type;
                // 对于非流式输出的消息，解析 Markdown 内容
                if (type === 'assistant' && !streamChecked.checked) {
                    div.innerHTML = marked.parse(message);
                } else if (type === 'user' || type === 'system') {
                    // 用户消息和系统消息保持纯文本
                    div.textContent = message;
                } else {
                    // 流式输出时保留原始文本，由 ws.onmessage 处理
                    div.textContent = message;
                }
                messages.appendChild(div);
                return div;
            }
    
            function clearChat() {
                messages.innerHTML = '';
                currentStreamDiv = null;
                currentStreamContent = '';
            }
            
            // 显示当前session_id状态
            window.addEventListener('load', function() {
                if (session_id) {
                    addMessage(`已恢复会话: ${session_id}`, 'system');
                } else {
                    addMessage('开始新会话', 'system');
                }
            });
        </script>
    </body>
    </html>
    """)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8011)
