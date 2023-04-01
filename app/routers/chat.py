"""API router for chatGPT"""
import asyncio
import logging
import re
import subprocess

from fastapi import APIRouter, WebSocket

router = APIRouter()

logger = logging.getLogger('uvicorn')


# Response from alpaca includes styling escape codes.
def remove_ansi_escape_codes(text):
    ansi_escape_pattern = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape_pattern.sub("", text)


async def execute_command(websocket):
    process = subprocess.Popen(
        ['/chat/alpaca.cpp/chat', '-m', '/models/ggml-alpaca-7b-q4.bin'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # auto encode/decode as utf-8
        bufsize=1,
    )

    # Receive chat AI's response
    async def read_stream(stream, callback):
        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None,
                stream.readline)
            if not line:
                break
            cleaned_line = remove_ansi_escape_codes(line)
            prefix_removed = cleaned_line.strip('> ')
            await callback(prefix_removed)

    # Log stderr output from chat AI
    async def log_stream(stream):
        while True:
            line = await asyncio.get_event_loop().run_in_executor(
                None,
                stream.readline)
            if not line:
                break
            logger.error(
                f'stderr from chat AI: {line.strip()}')

    # Tell user's message to chat AI
    async def write_stream():
        try:
            while True:
                data = await websocket.receive_text()
                if not data:
                    break
                process.stdin.write(f'{data}\n')
                process.stdin.flush()
        finally:
            # Kill the chat AI process when websocket is closed.
            process.kill()

    stdout_callback = lambda x: websocket.send_text(f"{x.strip()}")
    # stderr_callback = lambda x: websocket.send_text(f"error: {x.strip()}")

    await asyncio.gather(
        read_stream(process.stdout, stdout_callback),
        # read_stream(process.stderr, stderr_callback),
        log_stream(process.stderr),
        write_stream(),
    )


@router.websocket('/chat-gpt/ws')
async def chat_gpt(websocket: WebSocket):
    await websocket.accept()
    await execute_command(websocket)
    await websocket.close()
