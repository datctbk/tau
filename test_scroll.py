import asyncio
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.data_structures import Point

text_parts = ["Line 1\n"]

def get_text():
    return ANSI("".join(text_parts))

def get_cursor():
    lines = sum(s.count("\n") for s in text_parts)
    return Point(x=0, y=lines)

out_win = Window(content=FormattedTextControl(get_text, get_cursor_position=get_cursor), wrap_lines=True)
in_buf = Buffer()
in_win = Window(content=BufferControl(buffer=in_buf), height=1)

app = Application(layout=Layout(HSplit([out_win, in_win]), focused_element=in_win), full_screen=True)

async def stream():
    for i in range(2, 50):
        await asyncio.sleep(0.1)
        text_parts.append(f"Line {i}\n")
        app.invalidate()
    app.exit()

async def main():
    asyncio.create_task(stream())
    await app.run_async()

asyncio.run(main())
