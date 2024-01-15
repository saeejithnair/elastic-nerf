import uuid
from typing import Optional

import plotly.io as pio
from IPython.display import HTML, display


# This function takes a figure and displays it with a fullscreen button
def display_figure_fullscreen(fig, fig_name: Optional[str] = None):
    # Generate a random id for the iframe to avoid conflicts
    iframe_id = str(uuid.uuid4())

    # Save figure to a standalone HTML
    if fig_name is None:
        fig_name = iframe_id
    html_file = f"{fig_name}_figure.html"
    pio.write_html(fig, html_file)

    # SVG for fullscreen button
    svg_icon = """
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="24px"
        height="24px"
        viewBox="0 0 64 64"
        fill="#000000"
        stroke="#000000"
    >
        <polyline points="20 8 8 8 8 20"/>
        <line x1="8" y1="8" x2="24" y2="24"/>
        <polyline points="56 20 56 8 44 8"/>
        <line x1="56" y1="8" x2="40" y2="24"/>
        <polyline points="44 56 56 56 56 44"/>
        <line x1="56" y1="56" x2="40" y2="40"/>
        <polyline points="8 44 8 56 20 56"/>
        <line x1="8" y1="56" x2="24" y2="40"/>
    </svg>
    """

    # Now create a custom HTML string with your desired functionality
    html_string = f"""
    <div style="position: relative; width: 100%; height: 600px;">
        <iframe id="{iframe_id}" src="{html_file}" width="100%" height="100%" frameBorder="0"></iframe>
        <button style="position: absolute; top: 10px; left: 10px; background-color: transparent; border: none; cursor: pointer; padding: 5px; z-index: 10;" onclick="goFullscreen('{iframe_id}'); return false">{svg_icon}</button>
    </div>
    <script type="text/javascript">
        function goFullscreen(id) {{
            var element = document.getElementById(id);
            if (element.mozRequestFullScreen) {{
              element.mozRequestFullScreen();
            }} else if (element.webkitRequestFullScreen) {{
              element.webkitRequestFullScreen();
            }} else if (element.msRequestFullscreen) {{
              element.msRequestFullscreen();
            }}
        }}
    </script>
    """

    # Display the HTML string
    display(HTML(html_string))
