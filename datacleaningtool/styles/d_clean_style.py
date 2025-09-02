def get_css(page_bg_css_property: str) -> str:
    """
    Retourneert de volledige CSS-string voor de d_Clean.py pagina.
    De achtergrond-CSS wordt dynamisch ingevoegd.
    """
    return f"""
        <style>
            /* --- ALGEMENE PAGINA STIJL --- */
            .stApp {{ 
                {page_bg_css_property} 
                color: white !important;
            }}
            /* Algemene tekst elementen standaard wit */
            h1, h2, h3, h4, h5, h6, p, label, div[data-testid="stMarkdownContainer"] p, 
            div[data-testid="stText"], .stMarkdown p,
            [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{ 
                color: white !important; 
            }}

            /* --- KNOPPEN --- */
            .stButton>button {{
                color: white !important; background-color: #222222 !important;
                border: 1px solid #5A5A5A !important; border-radius: 8px !important;
                padding: 10px 24px !important; font-weight: bold !important;
                transition: background-color 0.2s ease-in-out, transform 0.1s ease !important;
            }}
            .stButton>button:hover {{
                background-color: #3C3C3C !important; border-color: #7A7A7A !important;
                transform: scale(1.03);
            }}
            .stButton>button:active {{ background-color: #111111 !important; transform: scale(0.98); }}
            .stButton>button[kind="primary"] {{ 
                background-color: #B71C1C !important; 
                border-color: #D32F2F !important;
            }}
            .stButton>button[kind="primary"]:hover {{ background-color: #D32F2F !important; }}

            /* --- PAGINA HEADER & LOGO --- */
            .page-header-container {{ text-align: center; margin-bottom: 20px; }}
            .page-header-container h1 {{ 
                font-weight: 700; font-size: 2.8rem;
                text-shadow: 2px 2px 8px rgba(0,0,0,0.6); letter-spacing: 0.03em; padding-bottom: 8px;
            }}
            .logo-container {{ display: flex; justify-content: center; align-items: center; padding-top: 20px; margin-bottom: 10px; }}
            .logo-image {{ width: 200px; max-width: 60%; height: auto; }}

            /* --- CHAT INTERFACE (MET FIX VOOR LIJSTJES) --- */
            div[data-testid="stChatMessage"] {{ 
                background-color: rgba(25, 25, 25, 0.9) !important; /* Iets minder transparant voor beter contrast */
                border-radius: 12px !important; border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0px 3px 6px rgba(0,0,0,0.35); margin-bottom: 12px !important;
            }}
            /* Regel voor alle paragrafen, lijstjes en algemene tekst in de chat bubble */
            div[data-testid="stChatMessage"] p,
            div[data-testid="stChatMessage"] ul li {{
                color: #E0E0E0 !important; /* Een heldere, lichtgrijze kleur */
                line-height: 1.6;
            }}
            /* Regel voor vetgedrukte tekst, maak deze puur wit voor nadruk */
            div[data-testid="stChatMessage"] strong, div[data-testid="stChatMessage"] b {{ 
                color: #FFFFFF !important;
                font-weight: 600;
            }}
            /* Extra: Styling voor `code` blokjes die de AI kan sturen */
            div[data-testid="stChatMessage"] code {{
                color: #FFD54F !important; /* Een lichte, warme gele kleur voor code */
                background-color: rgba(0,0,0,0.4) !important;
                padding: 2px 6px;
                border-radius: 5px;
                font-family: "Source Code Pro", monospace;
            }}

            div[data-testid="stChatInput"] textarea, .stChatFloatingInputContainer textarea {{
                background-color: rgba(230, 230, 230, 0.95) !important; color: black !important;
                border-radius: 8px !important; border: 1px solid #B0B0B0 !important;
            }}
            div[data-testid="stChatInput"] textarea::placeholder {{ color: #555555 !important; }}

            /* --- TABS & EXPANDERS --- */
            .stTabs [data-baseweb="tab-list"] {{ border-bottom: 2px solid #B71C1C; }}
            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                background-color: rgba(0,0,0,0.6); color: white; font-weight: bold;
            }}
            div[data-testid="stTabContent"] {{ 
                background-color: rgba(10, 10, 10, 0.5) !important; 
                border-radius: 0 0 8px 8px; padding: 20px;
                border: 1px solid rgba(183,28,28,0.3); 
            }}
            .stExpander header {{ 
                background-color: black !important; 
                color: white !important; 
                border-radius: 8px !important; 
                border-bottom: 1px solid #444444 !important; 
            }}
            div[data-testid="stExpanderDetails"] {{
                background-color: #1E1E1E !important; 
                color: white !important; 
                padding: 1.5em !important;
                border: 1px solid #444444 !important; 
                border-top: none !important; 
                border-radius: 0 0 8px 8px;
            }}

            /* Tekst en inputs BINNEN donkere expanders */
            div[data-testid="stExpanderDetails"] * {{
                color: white !important;
            }}
            div[data-testid="stExpanderDetails"] .stTextInput input,
            div[data-testid="stExpanderDetails"] .stTextArea textarea,
            div[data-testid="stExpanderDetails"] .stSelectbox div[data-baseweb="select"] > div {{
                background-color: #333333 !important; 
                color: white !important;
                border: 1px solid #555555 !important;
            }}

            /* Uitzondering voor JSON "boxen" in rapporten */
            div[data-testid="stExpanderDetails"] div[data-testid="stJson"], 
            div[data-testid="stJson"] 
            {{
                background-color: #FFFFFF !important; 
                color: black !important; 
                border: 1px solid #DCDCDC !important; 
                border-radius: 6px !important;
            }}
            div[data-testid="stExpanderDetails"] div[data-testid="stJson"] pre,
            div[data-testid="stJson"] pre {{ color: black !important; }}

            /* Toast Notificaties */
            [data-testid="stToast"] {{
                background-color: black !important; color: white !important;
                border: 1px solid #444444 !important; border-radius: 8px !important; 
            }}
            [data-testid="stToast"] * {{
                color: white !important;
            }}
        </style>
    """