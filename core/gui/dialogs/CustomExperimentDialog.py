import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class CustomExperimentDialog(simpledialog.Dialog):
    """Диалог для ввода пользовательских параметров опыта"""

    def __init__(self, parent, title, param_names):
        self.param_names = param_names
        self.result = None
        super().__init__(parent, title)

    def body(self, frame):
        self.entries = {}

        for i, param_name in enumerate(self.param_names):
            ttk.Label(frame, text=f"{param_name}:").grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(frame, width=15)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[param_name] = entry

        return list(self.entries.values())[0] if self.entries else None

    def validate(self):
        try:
            self.result = {}
            for param_name, entry in self.entries.items():
                value = entry.get().strip()
                if not value:
                    messagebox.showerror("Ошибка", f"Введите значение для параметра '{param_name}'!")
                    return False
                self.result[param_name] = float(value)
            return True
        except ValueError:
            messagebox.showerror("Ошибка", "Все значения должны быть числами!")
            return False