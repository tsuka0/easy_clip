import tkinter as tk
from ui.main_window import MainWindow

def main():

    root = tk.Tk()
    root.geometry("1400x900")

    app = MainWindow(root)

    root.mainloop()

if __name__ == "__main__":
    main()
