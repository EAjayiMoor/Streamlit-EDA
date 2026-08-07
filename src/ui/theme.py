import streamlit as st


def inject_moorhouse_theme() -> None:
    """Apply the shared Moorhouse visual foundation to the Streamlit shell."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

        :root {
            --mh-brand: #3c1053;
            --mh-brand-purple: #5c068c;
            --mh-accent: #00ab8e;
            --mh-ocean: #186e7e;
            --mh-orange: #e48949;
            --mh-muted: #bdb6b9;
            --mh-background: #fbfafb;
            --mh-foreground: #181018;
            --mh-surface: #ffffff;
            --mh-surface-muted: #f5f3f6;
            --mh-border: rgba(60, 16, 83, 0.10);
            --mh-border-strong: rgba(60, 16, 83, 0.20);
            --mh-text-muted: #71717a;
        }

        html, body, [class*="css"], .stApp {
            font-family: Poppins, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--mh-foreground);
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: var(--mh-background);
        }

        [data-testid="stSidebar"] {
            background: var(--mh-surface-muted);
            border-right: 1px solid var(--mh-border);
        }

        [data-testid="stSidebar"] * {
            font-family: inherit;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--mh-foreground);
            font-size: 0.875rem;
        }

        h1, h2, h3, h4 {
            color: var(--mh-brand);
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        h1 { font-size: 1.75rem; line-height: 1.15; }
        h2 { font-size: 1.25rem; line-height: 1.2; }
        h3 { font-size: 1.125rem; line-height: 1.25; }

        p, li, label, [data-testid="stCaptionContainer"] {
            font-size: 0.875rem;
            line-height: 1.45;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--mh-text-muted);
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            color: var(--mh-foreground);
            font-weight: 500;
        }

        button, input, textarea, [role="combobox"] {
            font-family: inherit !important;
        }

        button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: var(--mh-brand) !important;
            border: 1px solid var(--mh-brand) !important;
            border-radius: 6px !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background: var(--mh-brand-purple) !important;
            border-color: var(--mh-brand-purple) !important;
        }

        button[kind="secondary"],
        button[data-testid="baseButton-secondary"] {
            background: transparent !important;
            border: 1px solid var(--mh-border-strong) !important;
            border-radius: 6px !important;
            color: var(--mh-brand) !important;
            font-weight: 500 !important;
        }

        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="combobox"]:focus-visible {
            outline: 2px solid var(--mh-accent) !important;
            outline-offset: 2px !important;
        }

        input, textarea, [data-baseweb="select"] > div {
            background: var(--mh-surface) !important;
            border-color: var(--mh-border-strong) !important;
            border-radius: 6px !important;
        }

        [data-baseweb="select"] > div:focus-within {
            border-color: var(--mh-foreground) !important;
            box-shadow: 0 0 0 1px var(--mh-foreground) !important;
        }

        [data-testid="stTabs"] [role="tab"] {
            color: var(--mh-text-muted);
            font-size: 0.875rem;
            font-weight: 500;
            border-bottom: 2px solid transparent;
        }

        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--mh-brand);
            border-bottom-color: var(--mh-accent);
        }

        [data-testid="stMetric"] {
            background: var(--mh-surface);
            border: 1px solid var(--mh-border);
            border-radius: 8px;
            padding: 16px;
        }

        [data-testid="stMetricLabel"] {
            color: var(--mh-text-muted);
            font-weight: 500;
        }

        [data-testid="stMetricValue"] {
            color: var(--mh-brand);
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--mh-border);
            border-radius: 8px;
        }

        [data-testid="stExpander"] {
            background: var(--mh-surface);
            border: 1px solid var(--mh-border);
            border-radius: 8px;
        }

        [data-testid="stAlert"] {
            border-radius: 6px;
            font-family: inherit;
        }

        [data-testid="stAlert"][kind="info"] {
            background: var(--mh-surface-muted);
            border-left: 4px solid var(--mh-accent);
            color: var(--mh-foreground);
        }

        [data-testid="stAlert"][kind="success"] {
            border-left: 4px solid #166534;
        }

        [data-testid="stAlert"][kind="warning"] {
            border-left: 4px solid #92400e;
        }

        [data-testid="stAlert"][kind="error"] {
            border-left: 4px solid #991b1b;
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                transition-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
