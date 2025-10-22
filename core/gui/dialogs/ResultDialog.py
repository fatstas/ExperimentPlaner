import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class ResultDialog(simpledialog.Dialog):
    """Диалог для ввода результатов"""

    def __init__(self, parent, title, initial_result="", initial_notes=""):
        self.result = None
        self.initial_result = initial_result
        self.initial_notes = initial_notes
        super().__init__(parent, title)

    def body(self, frame):
        ttk.Label(frame, text="Результат:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.result_entry = ttk.Entry(frame, width=20)
        self.result_entry.grid(row=0, column=1, padx=5, pady=5)
        self.result_entry.insert(0, self.initial_result)

        ttk.Label(frame, text="Примечания:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.notes_text = tk.Text(frame, width=30, height=4)
        self.notes_text.grid(row=1, column=1, padx=5, pady=5)
        self.notes_text.insert(1.0, self.initial_notes)

        return self.result_entry

    def validate(self):
        result = self.result_entry.get().strip()
        notes = self.notes_text.get(1.0, tk.END).strip()

        if not result:
            messagebox.showerror("Ошибка", "Введите значение результата!")
            return False

        self.result = (result, notes)
        return True
