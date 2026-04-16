import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from src.search_service import ALGORITHMS, search_documents


class SearchEngineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Search Engine")
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.algorithm_var = tk.StringVar(value="splade")
        self.query_var = tk.StringVar()
        self.top_k_var = tk.IntVar(value=10)
        self.status_var = tk.StringVar(value="Ready")

        self._build_styles()
        self._build_layout()

    def _build_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Header.TLabel", font=("TkDefaultFont", 16, "bold"))
        style.configure("Subtle.TLabel", foreground="#555555")

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=18)
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(container)
        header.pack(fill=tk.X, pady=(0, 14))

        ttk.Label(header, text="Search Engine Demo", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(
            header,
            text="Choose an algorithm, enter a query, and inspect ranked passages.",
            style="Subtle.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        controls = ttk.LabelFrame(container, text="Search")
        controls.pack(fill=tk.X, pady=(0, 14))

        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Algorithm").grid(row=0, column=0, sticky=tk.W, padx=10, pady=(10, 6))
        algorithm_box = ttk.Combobox(
            controls,
            textvariable=self.algorithm_var,
            values=ALGORITHMS,
            state="readonly",
            width=18,
        )
        algorithm_box.grid(row=0, column=1, sticky=tk.W, padx=10, pady=(10, 6))

        ttk.Label(controls, text="Top K").grid(row=0, column=2, sticky=tk.W, padx=(20, 10), pady=(10, 6))
        top_k_spin = ttk.Spinbox(controls, from_=1, to=100, textvariable=self.top_k_var, width=8)
        top_k_spin.grid(row=0, column=3, sticky=tk.W, padx=10, pady=(10, 6))

        ttk.Label(controls, text="Query").grid(row=1, column=0, sticky=tk.W, padx=10, pady=(6, 10))
        query_entry = ttk.Entry(controls, textvariable=self.query_var)
        query_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=10, pady=(6, 10))
        query_entry.bind("<Return>", lambda event: self.start_search())

        self.search_button = ttk.Button(controls, text="Search", command=self.start_search)
        self.search_button.grid(row=1, column=3, sticky=tk.E, padx=10, pady=(6, 10))

        results_panel = ttk.LabelFrame(container, text="Results")
        results_panel.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(results_panel, wrap=tk.WORD, height=20, borderwidth=0)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)

        scrollbar = ttk.Scrollbar(results_panel, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        self.results_text.configure(state=tk.DISABLED)

        footer = ttk.Frame(container)
        footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(footer, textvariable=self.status_var, style="Subtle.TLabel").pack(anchor=tk.W)

        query_entry.focus_set()

    def start_search(self):
        query = self.query_var.get().strip()
        if not query:
            messagebox.showwarning("Missing query", "Please enter a search query.")
            return

        try:
            top_k = int(self.top_k_var.get())
        except (TypeError, ValueError):
            messagebox.showwarning("Invalid top_k", "Top K must be a positive integer.")
            return

        if top_k < 1:
            messagebox.showwarning("Invalid top_k", "Top K must be at least 1.")
            return

        algorithm = self.algorithm_var.get().strip().lower()
        self._set_busy(True)
        self.status_var.set(f"Running {algorithm.upper()} search... This can take a few minutes...")
        self._write_results(f"Searching with {algorithm.upper()} for: {query!r}\n\n")

        worker = threading.Thread(
            target=self._run_search,
            args=(algorithm, query, top_k),
            daemon=True,
        )
        worker.start()

    def _run_search(self, algorithm, query, top_k):
        started = time.time()
        try:
            results = search_documents(algorithm, query, top_k=top_k, log_search=True)
            elapsed_ms = (time.time() - started) * 1000
            self.root.after(
                0,
                lambda: self._show_results(algorithm, query, top_k, results, elapsed_ms),
            )
        except Exception as exc:
            self.root.after(0, lambda: self._show_error(exc))

    def _show_results(self, algorithm, query, top_k, results, elapsed_ms):
        lines = [
            f"Algorithm: {algorithm.upper()}",
            f"Query: {query}",
            f"Elapsed: {elapsed_ms:.0f} ms",
            "",
        ]

        if not results:
            lines.append("No results found.")
        else:
            for index, result in enumerate(results, start=1):
                snippet = result.get("text", "").replace("\n", " ").strip()
                if len(snippet) > 260:
                    snippet = f"{snippet[:260]}..."
                lines.append(f"[{index}]")
                lines.append(snippet)
                lines.append("")

        self._write_results("\n".join(lines))
        self.status_var.set(f"{algorithm.upper()} search completed in {elapsed_ms:.0f} ms")
        self._set_busy(False)

    def _show_error(self, exc):
        self._write_results("Search failed.\n")
        self.status_var.set("Search failed")
        self._set_busy(False)
        messagebox.showerror("Search error", str(exc))

    def _write_results(self, text):
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.configure(state=tk.DISABLED)

    def _set_busy(self, busy):
        state = tk.DISABLED if busy else tk.NORMAL
        self.search_button.configure(state=state)


def launch_app():
    if tk is None or ttk is None or messagebox is None:
        raise RuntimeError(
            "Tkinter is not available in this Python environment. "
            "Install the Tk runtime libraries before launching the GUI."
        )

    root = tk.Tk()
    SearchEngineApp(root)
    root.mainloop()


def main():
    launch_app()


if __name__ == "__main__":
    main()