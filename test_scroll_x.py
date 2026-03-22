import asyncio
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.data_structures import Point

def get_text():
    return "Line 1\nLine 2 is very long "

def get_cursor():
    return Point(x=999999, y=1)

out_win = Window(content=FormattedTextControl(get_text, get_cursor_position=get_cursor), wrap_lines=True)
app = Application(layout=Layout(out_win))

async def main():
    asyncio.create_task(stream())
    await app.run_async()

async def stream():
    await asyncio.sleep(0.1)
    app.exit()

asyncio.run(main())
