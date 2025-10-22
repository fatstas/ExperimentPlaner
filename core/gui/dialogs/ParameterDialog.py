import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class ParameterDialog(simpledialog.Dialog):
    """Диалог для ввода параметров"""

    def __init__(self, parent, title, initial_values=None):
        self.result = None
        self.initial_values = initial_values or ("", "", "")
        super().__init__(parent, title)

    def body(self, frame):
        ttk.Label(frame, text="Название параметра:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_entry = ttk.Entry(frame, width=20)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Нижняя граница:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.low_entry = ttk.Entry(frame, width=20)
        self.low_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Верхняя граница:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.high_entry = ttk.Entry(frame, width=20)
        self.high_entry.grid(row=2, column=1, padx=5, pady=5)

        if self.initial_values:
            self.name_entry.insert(0, self.initial_values[0])
            self.low_entry.insert(0, str(self.initial_values[1]))
            self.high_entry.insert(0, str(self.initial_values[2]))

        return self.name_entry

    def validate(self):
        try:
            name = self.name_entry.get().strip()
            low = float(self.low_entry.get())
            high = float(self.high_entry.get())

            if not name:
                messagebox.showerror("Ошибка", "Введите название параметра!")
                return False

            if low >= high:
                messagebox.showerror("Ошибка", "Нижняя граница должна быть меньше верхней!")
                return False

            self.result = (name, low, high)
            return True

        except ValueError:
            messagebox.showerror("Ошибка", "Границы должны быть числами!")
            return False